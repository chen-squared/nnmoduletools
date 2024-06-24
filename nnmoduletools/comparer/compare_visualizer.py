
import torch
import numpy as np

import matplotlib.pyplot as plt
import math
import warnings

from tqdm import tqdm

from multiprocessing import Pool
from pathlib import Path

from functools import lru_cache

###########################
# Colored Print Functions #
###########################

def color_str(text: str, color: str | None = None) -> str:
    if color is None or color.lower() == 'none':
        return text
    color_code_dict = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'purple': 35,
        'cyan': 36,
        'white': 37,
    }
    color_code = color_code_dict[color.lower()]
    return f"\033[{color_code}m{text}\033[0m"


##########################
# NPZ Compare Visualizer #
##########################

default_tol = {'f32': {'abs_tol': 2 ** -126, 'rel_tol': 2 ** -23},
               'f16': {'abs_tol': 2 ** -14, 'rel_tol': 2 ** -10},
               'bf16': {'abs_tol': 2 ** -126, 'rel_tol': 2 ** -7},
               'int8': {'abs_tol': 0., 'rel_tol': 0.}}

class NPZErrWrapper:
    def __init__(self, npz, role):
        assert isinstance(npz, np.lib.npyio.NpzFile)
        assert role in ['desired', 'actual']
        self.role = role
        self.npz = npz
        self._valid_keys = None

    @property
    def valid_keys(self):
        if self._valid_keys is None:
            self._valid_keys = {name[:-len(self.role)-1] for name in self.npz.keys() if name.endswith(self.role)}
        assert isinstance(self._valid_keys, set)
        return self._valid_keys

    def keys(self):
        return self.valid_keys

    def values(self):
        return (f"{name}_{self.role}" for name in self._valid_keys)

    def items(self):
        return zip(self.valid_keys, self.values())

    def __contains__(self, item):
        return item in self.valid_keys

    def __iter__(self):
        return iter(self.valid_keys())

    def __getitem__(self, key):
        return self.npz[f"{key}_{self.role}"]

    def __setitem__(self, key, value):
        self.npz[f"{key}_{self.role}"] = value

    def __len__(self):
        return len(self.valid_keys)

def err_comparer(fn):
    if isinstance(fn, str):
        npz = np.load(fn)
    elif isinstance(fn, np.lib.npyio.NpzFile):
        npz = fn
    else:
        assert 0, "Should provide file name or npz object!"
    target = NPZErrWrapper(npz, 'actual')
    ref = NPZErrWrapper(npz, 'desired')
    return NPZComparer(target, ref)


def model_tpu_comparer(fn):
    assert isinstance(fn, str) and fn.endswith(".npz"), "Please provide the file name of npz file!"
    if "_model_" in fn:
        pattern = fn.replace("_model_", "%s")
    elif "_tpu_" in fn:
        pattern = fn.replace("_tpu_", "%s")
    else:
        assert 0, "Should provide either model_out or tpu_out npz file!"
    return NPZComparer(pattern % "_model_", pattern % "_tpu_")


def get_nchw(darray, mix_axis):
    shape = darray.shape
    dims = len(shape)
    reshaped = [1, 1, 1, 1]
    if mix_axis is None:
        if dims > 4:
            for i in range(2):
                reshaped[- i - 1] = shape[- i - 1]
            for i in range(1, dims - 2):
                reshaped[1] *= shape[i]
            reshaped[0] = shape[0]
        else:
            for i in range(dims):
                reshaped[- i - 1] = shape[- i - 1]
    else:
        assert dims - len(mix_axis) + 1 == 4
        dim = mix_axis[0]
        for i in range(1, len(mix_axis)):
            assert mix_axis[i] - mix_axis[i - 1] == 1
        for i in mix_axis:
            reshaped[dim] *= shape[i]
        for i in range(dim):
            reshaped[i] = shape[i]
        for i in range(dims - dim - len(mix_axis)):
            reshaped[3 - i] = shape[dims - i - 1]
    return reshaped


def recalc_hw(h, w):
    all_hw = h * w
    for i in range(int(math.sqrt(all_hw)), 0, -1):
        if all_hw % i == 0:
            return all_hw // i, i


def assign_hw(h, w, new_hw):
    n_ele = h * w
    h1, w1 = new_hw
    if h1 == -1 and w1 == -1:
        return math.ceil(np.sqrt(n_ele)), math.ceil(np.sqrt(n_ele))
    elif h1 == -1 and w1 != -1:
        return math.ceil(n_ele / w1), w1
    elif h1 != -1 and w1 == -1:
        return h1, math.ceil(n_ele / h1)
    else:
        assert h1 * w1 >= n_ele
        return h1, w1


def assign_new_shape(reshaped, resize_hw):
    h, w = reshaped[2], reshaped[3]
    if isinstance(resize_hw, str):
        if resize_hw.lower() == "rectangle" or resize_hw.lower() == "auto":
            reshaped[2], reshaped[3] = recalc_hw(h, w)
        elif resize_hw.lower() == 'square':
            reshaped[2], reshaped[3] = assign_hw(h, w, (-1, -1))
        elif resize_hw.lower() == "none":
            pass
        else:
            assert 0, "Invalid param resize_hw=%s" % resize_hw
    elif isinstance(resize_hw, int):
        assert resize_hw > 0
        reshaped[2], reshaped[3] = assign_hw(h, w, (resize_hw, -1))
    elif isinstance(resize_hw, (tuple, list)):
        assert len(resize_hw) == 2
        reshaped[2], reshaped[3] = assign_hw(h, w, resize_hw)
    elif resize_hw is None:
        pass
    else:
        assert 0, "Invalid param resize_hw=%s" % resize_hw
    return np.array([1] * (h * w) + [0] * (reshaped[2] * reshaped[3] - h * w)).reshape((reshaped[2], reshaped[3]))


def get_data_dist(darray, data_mask):
    to_calculate = darray[data_mask == 1]
    to_calculate = to_calculate[~np.isinf(to_calculate)]
    try:
        real_mean = np.nanmean(to_calculate)
        real_min = np.nanmin(to_calculate)
        real_max = np.nanmax(to_calculate)
    except ValueError:
        real_mean = 0.
        real_min = 0.
        real_max = 0.
    
    real_mean = 0. if np.isnan(real_mean) else real_mean
    real_min = 0. if np.isnan(real_min) else real_min
    real_max = 0. if np.isnan(real_max) else real_max
    return real_mean, real_min, real_max


def plot_2d_array(diff, data_mask=None, title="", figsize=6, vmin=-0.1, vmax=0.1, h_split=None, w_split=None):
    figwidth = figsize
    figheight = 3 + figsize / diff.shape[1] * diff.shape[0]
    plt.figure(figsize=(figwidth, figheight))
    nan_mask = np.where(np.isnan(diff).astype(float) * data_mask, 1, np.nan)
    pos_inf_mask = np.where(np.isposinf(diff).astype(float) * data_mask, 1, np.nan)
    neg_inf_mask = np.where(np.isneginf(diff).astype(float) * data_mask, 1, np.nan)
    plt.imshow(nan_mask, 'Greys', vmin=0, vmax=1.25)
    plt.imshow(pos_inf_mask, 'Oranges', vmin=0, vmax=2)
    plt.imshow(neg_inf_mask, 'Greens', vmin=0, vmax=1.5)
    if not data_mask is None:
        # diff += np.where(data_mask == 0, np.nan, 0)
        diff[data_mask == 0] += np.nan
    plt.imshow(diff, 'bwr', vmin=vmin, vmax=vmax)
    plt.xlim(-2, diff.shape[1] + 1)
    plt.ylim(diff.shape[0] + 1, -2)
    ax = plt.gca()
    ax.set_facecolor('lightgrey')
    ax.xaxis.set_tick_params(which='major', top=True,
                             labeltop=True, labelbottom=True)
    ax.yaxis.set_tick_params(which='major', right=True,
                             labelleft=True, labelright=True)

    if h_split is None:
        h_split = max(1, int(diff.shape[0] / (figheight - 3) / 3 + 0.5))
        h_ticks = np.arange(0, diff.shape[0], h_split)
        h_range = range(0, diff.shape[0], h_split)
    else:
        h_ticks = np.arange((h_split - 1) / 2, diff.shape[0], h_split)
        h_range = range(diff.shape[0] // h_split)
    if w_split is None:
        w_split = max(1, int(diff.shape[1] / figwidth / 3 + 0.5))
        w_ticks = np.arange(0, diff.shape[1], w_split)
        w_range = range(0, diff.shape[1], w_split)
        rotation = 'vertical' if diff.shape[1] > 100 + w_split else None
    else:
        w_ticks = np.arange((w_split - 1) / 2, diff.shape[1], w_split)
        w_range = range(diff.shape[1] // w_split)
        rotation = 'vertical' if diff.shape[1] // w_split >= 100 else None
    plt.yticks(h_ticks, h_range)
    for i in range(h_split, diff.shape[0], h_split):
        plt.axhline(i - 0.5, 0, diff.shape[1], color='lightgrey')
    plt.xticks(w_ticks, w_range, rotation=rotation)
    for i in range(w_split, diff.shape[1], w_split):
        plt.axvline(i - 0.5, 0, diff.shape[0], color='lightgrey')
    title_len, title_line = len(title), figsize * 8
    plt.title(title if title_len <= title_line else "\n".join(
        [title[i: i + title_line] for i in range(0, title_len, title_line)]))
    # plt.colorbar()
    plt.show()


def make_slice_object(something):
    if isinstance(something, slice):
        return something
    elif isinstance(something, int):
        if something == -1:
            return slice(None)
        return slice(something, something + 1)
    elif isinstance(something, (list, tuple)):
        assert 1 <= len(something) <= 3
        return slice(*something)
    elif something is None:
        return None
    else:
        raise ValueError("Invalid slice param")


class NPZWrapper:
    def __init__(self, npz, role="darray"):
        if isinstance(npz, (str, Path)):
            self.npz = np.load(npz)
        elif isinstance(npz, (np.lib.npyio.NpzFile, dict, NPZErrWrapper)):
            self.npz = npz
        elif isinstance(npz, np.ndarray):
            self.npz = {'darray': npz}
        elif isinstance(npz, torch.Tensor):
            self.npz = {'darray': npz.detach().cpu().numpy()}
        else:
            raise TypeError
        self.role = role

    def keys(self):
        return self.npz.keys()

    def values(self):
        return self.npz.values()

    def items(self):
        return self.npz.items()

    def __contains__(self, item):
        return item in self.npz

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        return self.npz[key]

    def __setitem__(self, key, value):
        self.npz[key] = value

    def __len__(self):
        return len(self.npz)

    def info(self, tensor=None):
        if tensor is None:
            tensors = self.keys()
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys()
                tensors = tensor
            else:
                tensors = [tensor]
        for key in tensors:
            print("tensor='%s'," % key, "#shape", self[key].shape)

    def get_darray(self, tensor=None, slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None, print_shape=True):
        new_darray, data_mask, attr, print_shape_str = self._get_darray(tensor, slices, index, c_columns, resize_hw, transpose_hw, mix_axis)
        if print_shape:
            print(print_shape_str)
        return new_darray, data_mask, attr
    
    @lru_cache(maxsize=10)
    def get_data_dist(self, tensor=None, slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None):
        new_darray, data_mask, attr, print_shape_str = self._get_darray(tensor, slices, index, c_columns, resize_hw, transpose_hw, mix_axis)
        return get_data_dist(new_darray, data_mask)
    
    @lru_cache(maxsize=10)
    def _get_darray(self, tensor=None, slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None):
        if tensor is None:
            tensor = list(self.keys())[0]
        slice_list = tuple(make_slice_object(e)
                           for e in slices) if not slices is None else (slice(None),)
        if len(self[tensor].shape) == 0:
            darray = self[tensor].reshape(1).astype(float)
        else:
            darray = self[tensor][slice_list].astype(float)
        reshaped = get_nchw(darray, mix_axis)
        print_shape_str = ""
        if not slices is None:
            print_shape_str +="sliced "
        print_shape_str +="shape %s, reshaped to %s, " % (darray.shape, tuple(reshaped))
        darray = np.reshape(darray, reshaped)
        data_mask_channel = assign_new_shape(reshaped, resize_hw)
        if transpose_hw:
            darray = np.transpose(darray, [0, 1, 3, 2])
            if resize_hw in {'none', None, 'rectangle', 'auto'}:
                reshaped[2], reshaped[3] = reshaped[3], reshaped[2]
                data_mask_channel = np.reshape(
                    data_mask_channel.reshape(-1), data_mask_channel.shape[::-1])
            print_shape_str += "with hw transposed data "
        print_shape_str += "shown in %s" % (tuple(reshaped), )

        n_, c_, h_, w_ = reshaped
        if c_columns == -1: c_columns = c_
        per_c = math.ceil(c_ / c_columns)

        if not index is None:
            n, c = index
            new_darray = np.resize(
                darray[n // per_c][(n % per_c) * c_columns + c], data_mask_channel.shape)
            data_mask = data_mask_channel
            h_, w_ = None, None
        else:
            frame_shape = [n_ * per_c, min(c_, c_columns)]
            new_darray = np.block([[np.resize(darray[n // per_c][(n % per_c) * c_columns + c], data_mask_channel.shape) if 0 <= (n % per_c) *
                                    c_columns + c < c_ else np.zeros_like(data_mask_channel) for c in range(frame_shape[1])] for n in range(frame_shape[0])])
            data_mask = np.block([[data_mask_channel if 0 <= (n % per_c) *
                                   c_columns + c < c_ else np.zeros_like(data_mask_channel) for c in range(frame_shape[1])] for n in range(frame_shape[0])])

            # # trying to accelerate, but seems to be failed. suspended.
            
            # # method 1
            # frame_shape = [n_ * per_c * h_, min(c_, c_columns) * w_]
            # new_darray = np.zeros(frame_shape)
            # data_mask = np.zeros(frame_shape)
            # x, y = np.indices(new_darray.shape)
            # x_ok, y_ok = np.where((((x // h_) % per_c) * c_columns + y // w_ < c_ ) * ((x % h_) * w_  + (y % w_) < darray.shape[-2] * darray.shape[-1]))
            # n, c, h, w = (x_ok // (per_c * h_)), ((x_ok // h_) % per_c) * c_columns + y_ok // w_, x_ok % h_, y_ok % w_
            # data_mask[x_ok, y_ok] = data_mask_channel[h, w]
            # h, w = (h * w_ + w) // darray.shape[-1], (h * w_ + w) % darray.shape[-1]
            # new_darray[x_ok, y_ok] = darray[n, c, h, w]

            # # method 2
            # new_darray = np.zeros((frame_shape[0], frame_shape[1], data_mask_channel.shape[0] * data_mask_channel.shape[1]))
            # data_mask = np.zeros((frame_shape[0], frame_shape[1]) + data_mask_channel.shape)
            # n_indices, c_indices = np.indices(frame_shape)
            # valid_indices = np.where((n_indices % per_c) * c_columns + c_indices < c_)
            # new_darray[valid_indices[0], valid_indices[1], :darray.shape[-1] * darray.shape[-2]] = darray[valid_indices[0] // per_c, (valid_indices[0] % per_c) * c_columns + valid_indices[1]].reshape(-1, darray.shape[-1] * darray.shape[-2])
            # data_mask[valid_indices] = data_mask_channel
            # new_darray = new_darray.reshape(new_darray.shape[0], new_darray.shape[1], data_mask_channel.shape[0], data_mask_channel.shape[1]).transpose((0, 2, 1, 3)).reshape(new_darray.shape[0] * data_mask_channel.shape[0], new_darray.shape[1] * data_mask_channel.shape[1])
            # data_mask = data_mask.transpose((0, 2, 1, 3)).reshape(data_mask.shape[0] * data_mask.shape[2], data_mask.shape[1] * data_mask.shape[3])

        attr = {'title': ' '.join([tensor, str(slices) if not slices is None else "", str(index) if not index is None else ""]).strip(),
                'h_split': h_,
                'w_split': w_,
                }

        return new_darray, data_mask, attr, print_shape_str

    def plot(self, tensor=None, abs_tol=None, rel_tol=None, figsize=6, vmin=None, vmax=None, **kwargs):
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors = self.keys()
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys()
                tensors = tensor
            else:
                tensors = [tensor]

        for key in tensors:
            print("tensor='%s'," % key)
            darray, data_mask, attr = self.get_darray(key, **kwargs)
            real_mean, real_min, real_max = self.get_data_dist(key, **kwargs)
            print("data distribution: mean %s, min %s, max %s" %
                  (real_mean, real_min, real_max))

            if abs_tol is None:
                abs_tol_ = real_mean
            else:
                abs_tol_ = abs_tol
            if rel_tol is None:
                rel_tol_ = max(abs(real_max - abs_tol_),
                               abs(real_min - abs_tol_))
            else:
                rel_tol_ = rel_tol
            if rel_tol_ == 0.:
                rel_tol_ += 1E-10
            if vmin is None or vmax is None:
                vmin = abs_tol_ - rel_tol_ 
                vmax = abs_tol_ + rel_tol_  
            print(f"vmin {abs_tol_ - rel_tol_} zero point {abs_tol_} vmax {abs_tol_ + rel_tol_} ")

            _attr = {k: v for k, v in attr.items()}
            _attr['title'] = "%s: %s" % (self.role, attr['title'])
            plot_2d_array(darray, data_mask, figsize=figsize,
                          vmin=vmin, vmax=vmax, **_attr)


class NPZComparer:
    def __init__(self, target, ref):
        self.target = NPZWrapper(target, 'Target')
        self.ref = NPZWrapper(ref, 'Ref')
        self._keys = None
        self._error_keys = None
        self._isolated_keys = None
        self.archived_kwargs = {}

    # lazy initialization
    @property
    def keys(self):
        if self._keys is None and self._error_keys is None:
            self._keys = {}
            self._error_keys = {}
            for key in tqdm(self.ref, desc="Loading NPZ"):
                if key in self.target:
                    if self.target[key].size != self.ref[key].size:
                        print(f"Error: Tensor {key} shape not same: {self.target[key].shape} vs {self.ref[key].shape}.")
                        self._error_keys[key] = 1
                    else:
                        self._keys[key] = 1                   
            assert len(self._keys) > 0, 'No common data.'
        assert(isinstance(self._keys, dict) and isinstance(self._error_keys, dict)), "Keys not initialized!"
        return self._keys

    @property
    def error_keys(self):
        if self._keys is None and self._error_keys is None:
            self.keys
        assert(isinstance(self._keys, dict) and isinstance(self._error_keys, dict)), "Keys not initialized!"
        return self._error_keys
    
    @property
    def isolated_keys(self):
        if self._isolated_keys is None:
            self._isolated_keys = {
                "target": {key for key in self.target if key not in self.keys},
                "ref": {key for key in self.ref if key not in self.keys}
            }
        assert isinstance(self._isolated_keys, dict), "Isolated keys not initialized!"
        return self._isolated_keys
    
    def summary(self, show_isolated=False):
        print(f"Target tensors: {len(self.target)}, Ref tensors: {len(self.ref)}, Common tensors: {len(self.keys)}, Unmatched tensors:{len(self.error_keys)}")
        if show_isolated:
            if len(self.isolated_keys["target"]) > 0:
                print("Isolated target tensors:")
                for key in self.isolated_keys["target"]:
                    print("tensor='%s'," % key, "#shape", self.target[key].shape)
            if len(self.isolated_keys["ref"]) > 0:
                print("Isolated ref tensors:")
                for key in self.isolated_keys["ref"]:
                    print("tensor='%s'," % key, "#shape", self.ref[key].shape)
        
    def info(self, tensor=None, show_isolated=False):
        if tensor is None:
            tensors = self.keys
            self.summary(show_isolated=show_isolated)
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]
        for key in tensors:
            print("tensor='%s'," % key, "#shape", self.ref[key].shape)

    def compare(self, tolerance=(-1., -1.), tensor=None, verbose=1, summary=False):
        if tensor is None:
            tensors = self.keys
            if verbose:
                self.summary()
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]

        min_similarity = np.array([1.0, 1.0, np.inf])
        ALL_PASS = 1

        results = {}
        total = len(tensors)
        with Pool() as pool:
            with tqdm(total=total, desc="Comparing") as pbar:
                for key in tensors:
                    results[key] = pool.apply_async(calc_similarity,
                                                    args=(self.target[key].reshape(self.ref[key].shape), self.ref[key]),
                                                    callback=lambda x: pbar.update(1))
                pool.close()
                pool.join()
                pbar.close()

        for key in tensors:
            result = results[key].get()
            PASS = 1
            if 'similarity' in result:
                similarity = np.array(result['similarity'])
                min_similarity = np.minimum(min_similarity, similarity)
                if np.any(similarity[:2] < np.array(tolerance)):
                    PASS = 0
                    ALL_PASS = 0
            if verbose and not summary:
                print(color_str(f"tensor='{key}', #{self.ref[key].shape} {''.join(('%s: (%.6f, %.6f, %.6f)' % (k, *v) if isinstance(v, tuple) else '%s: %s' % (k, v)) for k, v in result.items())}{(' √' if PASS else ' ×') if tolerance != (-1., -1) else '' }", 'none' if tolerance == (-1., -1) else ('green' if PASS else 'red')))
                if verbose > 1 and not PASS:
                    self.dump_vs(tensor=key, c_columns=-1, top_k=10)
        if verbose:
            print(color_str(f"min_similarity: {tuple(min_similarity)}{(' √' if ALL_PASS else ' ×') if tolerance != (-1., -1.) else ''}", 'none' if tolerance == (-1., -1.) else ('green' if ALL_PASS else 'red')))
        return bool(ALL_PASS)

    def get_diff(self, tensor=None, abs_tol=0, rel_tol=0, **kwargs):
        if tensor is None:
            tensor = list(self.keys())[0]
        target, data_mask1, attr1 = self.target.get_darray(
            tensor, print_shape=False, **kwargs)
        ref, data_mask2, attr2 = self.ref.get_darray(
            tensor, print_shape=False, **kwargs)
        if target.shape != ref.shape:
            assert target.size == ref.size
            target = target.reshape(ref.shape)
            data_mask1 = data_mask1.reshape(data_mask2.shape)
            attr1 = attr2
        assert target.shape == ref.shape and np.all(
            data_mask1 == data_mask2) and attr1 == attr2
        compare = calc_similarity(target, ref, data_mask1)

        _attr = {k: v for k, v in attr1.items()}
        mask = 1 - np.isclose(target, ref, atol=abs_tol, rtol=rel_tol)
        if np.sum(mask) == 0:
            _attr['title'] += ' - No difference'
            warnings.warn("No difference under given tolerances.")

        diff = target - ref
        diff *= mask
        if rel_tol != 0:
            abs_ref = np.abs(ref)
            diff /= abs_ref + ((diff == 0) * (abs_ref == 0) if abs_tol == 0 else abs_tol / rel_tol)
        _attr['title'] = 'Diff: %s' % _attr['title']

        return diff, data_mask1, attr1, compare

    def plot_diff(self, tensor=None, abs_tol=0, rel_tol=0, figsize=6, vmin=None, vmax=None, **kwargs):
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors = self.keys
        else:
            if isinstance(tensor, list):
                for ts in tensor:
                    assert ts in self.keys
                tensors = tensor
            else:
                tensors = [tensor]
        print("abs tol %s, rel tol %s" % (abs_tol, rel_tol))
        for key in tensors:
            print("tensor='%s'," % key)
            darray, data_mask, attr, compare = self.get_diff(
                key, abs_tol=abs_tol, rel_tol=rel_tol,  **kwargs)
            real_mean, real_min, real_max = get_data_dist(darray, data_mask)
            abs_mean, abs_min, abs_max = get_data_dist(np.abs(darray), data_mask)
            print(f'Max of {"rel" if rel_tol != 0 else "abs"} diff: neg {real_min}, pos {real_max}, mean {real_mean}')
            print(f'Abs of {"rel" if rel_tol != 0 else "abs"} diff: min {abs_min}, max {abs_max}, mean {abs_mean}')
            print(*(f"{k}: {v}" for k, v in compare.items()))
            diff_max = max(abs(real_min if vmin is None else vmin), abs(real_max if vmax is None else vmax))
            if diff_max == 0:
                diff_max += 0.1
            vmin_, vmax_ = -diff_max, diff_max
            print("diffmin %s diffmax %s" % (vmin_, vmax_))
            plot_2d_array(darray, data_mask, figsize=figsize,
                          vmin=vmin_, vmax=vmax_, **attr)

    def plot(self, *args, **kwargs):
        self.plot_diff(*args, **kwargs)

    def plot_vs(self, tensor=None, abs_tol=1e-8, rel_tol=1e-3, figsize=6, diffmin=-0.1, diffmax=0.1, 
                zero_point=0.0, vmin=None, vmax=None,
                slices=None, index=None, c_columns=32, resize_hw=None, transpose_hw=False, mix_axis=None,
                dump=False, verbose=False):
        # archive the kwargs for next dump_vs
        self.archived_kwargs = {}
        self.archived_kwargs.update(
            tensor=tensor, abs_tol=abs_tol, rel_tol=rel_tol,
            slices=slices, index=index, c_columns=c_columns, resize_hw=resize_hw, transpose_hw=transpose_hw, mix_axis=mix_axis,
        )

        kwargs = {'slices': slices, 'index': index, 'c_columns': c_columns, 'resize_hw': resize_hw, 'transpose_hw': transpose_hw, 'mix_axis': mix_axis}
        tensors = []
        if tensor is None:
            warnings.warn(
                "Your are plotting all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors.extend(self.keys)
        else:
            if isinstance(tensor, list):
                for tensor in tensor:
                    assert tensor in self.keys
                    tensors.append(tensor)
            else:
                tensors.append(tensor)
        if not vmin is None or not vmax is None:
            warnings.warn(
                "vmin and vmax has changed to diffmin and diffmax. Now vmin and vmax affect plot_ref and plot_target.", Warning)
        for key in tensors:
            # target_darray, data_mask1, attr1 = self.target.get_darray(
            #     key, print_shape=False, **kwargs)
            # ref_darray, data_mask2, attr2 = self.ref.get_darray(
            #     key, print_shape=False, **kwargs)
            # assert np.all(data_mask1 == data_mask2) and attr1 == attr2
            target_mean, target_min, target_max = self.target.get_data_dist(key, **kwargs)
            ref_mean, ref_min, ref_max = self.ref.get_data_dist(key, **kwargs)
            zp = zero_point if not zero_point is None else (target_mean + ref_mean) / 2
            all_min = vmin if not vmin is None else min(target_min, ref_min)
            all_max = vmax if not vmax is None else max(target_max, ref_max)
            scale = max(abs(all_max - zp), abs(all_min - zp))
            if scale == 0:
                scale += 1E-10

            self.target.plot(key, abs_tol=zp, rel_tol=scale, figsize=figsize, **kwargs)
            self.ref.plot(key, abs_tol=zp, rel_tol=scale, figsize=figsize, **kwargs)
            self.plot_diff(key, abs_tol=abs_tol, rel_tol=rel_tol, figsize=figsize,
                           vmin=diffmin, vmax=diffmax, **kwargs)
        if dump:
            self.dump_vs_plot(verbose=verbose)

    def plot_ref(self, *args, **kwargs):
        self.ref.plot(*args, **kwargs)

    def plot_target(self, *args, **kwargs):
        self.target.plot(*args, **kwargs)

    def dump_vs_plot(self, abs_tol=None, rel_tol=None, verbose=None, top_k=None, **kwargs):
        archived_kwargs = self.archived_kwargs.copy()
        archived_kwargs.update(kwargs)
        if not abs_tol is None:
            archived_kwargs['abs_tol'] = abs_tol
        if not rel_tol is None:
            archived_kwargs['rel_tol'] = rel_tol
        if not verbose is None:
            archived_kwargs['verbose'] = verbose
        if not top_k is None:
            archived_kwargs['top_k'] = top_k
        self.dump_vs(**archived_kwargs)

    def dump_vs(self, tensor=None, abs_tol=1e-8, rel_tol=1e-3,
                slices=None, index=None, c_columns=-1, resize_hw=None, transpose_hw=False, mix_axis=None,
                verbose=False, top_k=None, **kwargs):
        kwargs_plot = {'slices': slices, 'index': index, 'c_columns': c_columns, 'resize_hw': resize_hw, 'transpose_hw': transpose_hw, 'mix_axis': mix_axis}
        tensors = []
        if tensor is None:
            warnings.warn(
                "Your are dumping all the tensors in the NPZ file. This may cause problems when the file is large.", Warning)
            tensors.extend(self.keys)
        else:
            if isinstance(tensor, list):
                for tensor in tensor:
                    assert tensor in self.keys
                    tensors.append(tensor)
            else:
                tensors.append(tensor)
        for key in tensors:
            print(f"tensor='{key}', # original shape {self.ref[key].shape}", end=", ")
            target_darray, data_mask1, attr1 = self.target.get_darray(
                key, print_shape=False, **kwargs_plot)
            ref_darray, data_mask2, attr2 = self.ref.get_darray(
                key, print_shape=True, **kwargs_plot)
            assert np.all(data_mask1 == data_mask2) and attr1 == attr2

            print(f"{''.join(('%s: (%.6f, %.6f, %.6f)' % (k, *v) if isinstance(v, tuple) else '%s: %s' % (k, v)) for k, v in calc_similarity(target_darray, ref_darray, data_mask1).items())}", end=", ")
            print("tolerance: abs %s, rel %s" % (abs_tol, rel_tol))
            if not top_k is None:
                print("top %s errors:" % top_k)
            diff = target_darray - ref_darray
            abs_ref = np.abs(ref_darray)
            zero_mask = ((diff == 0) * (abs_ref == 0)) if rel_tol == 0 or abs_tol == 0 else abs_tol / rel_tol
            rel_diff = diff / (abs_ref + zero_mask)
            error_magnitude = ((np.abs(diff) > 0) * 100 if abs_tol == 0 else np.abs(diff) / abs_tol) if rel_tol == 0 else np.abs(rel_diff) / rel_tol
            
            print("%20s %15s %15s %15s %15s" % ("index", "target", "ref", "abs_diff", "rel_diff"))
            
            if not top_k is None:
                top_k = min(top_k, np.sum(data_mask1))
                sorted_idx_x, sorted_idx_y = np.unravel_index(np.argsort(error_magnitude, axis=None), error_magnitude.shape)
                error_idx_hw = zip(sorted_idx_x[:-top_k-1:-1], sorted_idx_y[:-top_k-1:-1])
                error_idx_nchw_hw = [(idx_to_nchw(idx, attr1), idx) for idx in error_idx_hw]
            else:
                error_idx_hw = np.indices(error_magnitude.shape).reshape(2, -1).T
                error_idx_nchw_hw = sorted([(idx_to_nchw(idx, attr1), idx) for idx in error_idx_hw], key=lambda x: x[0])
                
            for (n, c, h, w), (h_, w_) in error_idx_nchw_hw:
                error_mark = 100 if np.isnan(error_magnitude[h_][w_]) else min(error_magnitude[h_][w_], 100)
                color = 'none' if diff[h_][w_] == 0 else ('blue' if diff[h_][w_] < 0 else 'red')
                if verbose or error_mark > 1:
                    print("%20s %#15.8g %#15.8g %s %s %s" % ((h, w) if attr1["h_split"] is None else (n, c, h, w),
                                                             target_darray[h_][w_],
                                                             ref_darray[h_][w_],
                                                             color_str("%#15.8g" % diff[h_][w_], 'green' if abs(diff[h_][w_]) <= abs_tol else color),
                                                             color_str("%#15.8g" % rel_diff[h_][w_], 'green' if abs(rel_diff[h_][w_]) <= rel_tol else color),
                                                             color_str("!" * int(error_mark), color)))
            print("")

def idx_to_nchw(idx, attr):
    if attr['w_split'] is None and attr['h_split'] is None:
        return (0, 0, *idx)
    n = idx[0] // attr['h_split']
    c = idx[1] // attr['w_split']
    h = idx[0] % attr['h_split']
    w = idx[1] % attr['w_split']
    return (n, c, h, w)

def nchw_to_idx(nchw, attr):
    n, c, h, w = nchw
    if attr['w_split'] is None and attr['h_split'] is None:
        return (h, w)
    return (n * attr['h_split'] + h, c * attr['w_split'] + w)

def close_order(x, y):
    y_ = y + (y == 0) * 1e-10
    rel_diff = np.nanmax((np.abs(x - y) - 1e-8) / np.abs(y_))
    if np.isnan(rel_diff):
        return 0
    order_raw = -np.log10(np.abs(rel_diff))
    return int(np.clip(order_raw, 0.0, 5.0))

# def square_rooted(x):
#     return np.sqrt(np.sum(np.power(x, 2)))
        
# def cosine_similarity(x, y):
#     numerator = np.sum(x * y)
#     sqrt_x = square_rooted(x)
#     sqrt_y = square_rooted(y)
#     denominator = sqrt_x * sqrt_y
#     if denominator == 0.0:
#         if sqrt_x == 0.0 and sqrt_y == 0.0:
#             return 1.0
#         else:
#             return 0.0
#     return numerator / denominator

# def euclidean_similarity(x, y):
#     ed = np.sqrt(np.sum(np.power(x - y, 2)))
#     sr = square_rooted((x + y) / 2) + 1e-7
#     if (np.isinf(ed) or np.isinf(sr)):
#         res = 0.0
#     else:
#         res = 1 - ed / sr
#     return res

# def sqnr_similarity(signal_raw, signal_dequant):
#     raw = signal_raw.ravel()
#     dequant = signal_dequant.ravel()

#     noise = raw - dequant

#     avg_raw = np.sum(raw) / raw.size
#     avg_noise = np.sum(noise) / noise.size

#     var_raw_zero_mean = np.sum(np.square(raw - avg_raw))
#     var_noise_zero_mean = np.sum(np.square(noise - avg_noise))
#     if var_noise_zero_mean == 0 or var_raw_zero_mean == 0:
#         return float('inf')
#     sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

#     return sqnr

def calc_similarity_opt(x, y):
    x_sum, y_sum = np.sum(x), np.sum(y)
    if np.isinf(x_sum) or np.isinf(y_sum):
        x, y = x.astype(np.float64), y.astype(np.float64)
        x_sum, y_sum = np.sum(x), np.sum(y)
    scale = np.max(np.abs([x_sum, y_sum])) / x.size
    if scale ** 2 * x.size > 3.4e38:
        x, y, x_sum, y_sum = x / scale, y / scale, x_sum / scale, y_sum / scale
    x_2, y_2 = np.linalg.norm(x) ** 2, np.linalg.norm(y) ** 2
    xy = np.dot(x, y)
    # cosine similarity
    cos_sim = 0.0
    denominator = np.sqrt(x_2 * y_2)
    if denominator == 0.0:
        if x_2 == 0.0 and y_2 == 0.0:
            cos_sim = 1.0
        else:
            cos_sim = 0.0
    else:
        cos_sim = xy / denominator
    # euclid similarity
    _2xy = 2 * xy
    ed = np.sqrt(np.abs(x_2 + y_2 - _2xy))
    sr = np.sqrt(x_2 + y_2 + _2xy) / 2 + 1e-7
    euc_sim = 0.0 if np.isinf(ed) or np.isinf(sr) else (1 - ed / sr)
    # sqnr similarity
    var_raw_zero_mean = max(x_2 - np.square(x_sum) / x.size, 0.0)
    var_noise_zero_mean = max((x_2 + y_2 - _2xy) - np.square(x_sum - y_sum) / x.size, 0.0)
    sqnr_sim = float('inf') if var_noise_zero_mean == 0 or var_raw_zero_mean == 0 else (10 * np.log10(var_raw_zero_mean / var_noise_zero_mean))
    return cos_sim, euc_sim, sqnr_sim

def calc_similarity(target, ref, mask=None):
    if mask is None:
        target_flatten = target.flatten().astype(float)
        ref_flatten = ref.flatten().astype(float)
    else:
        target_flatten = target[mask == 1].flatten().astype(float)
        ref_flatten = ref[mask == 1].flatten().astype(float)
    # compare similarity as npz_tool does
    if np.all(target_flatten == ref_flatten):
        return {"equal": "all equal"}

    close = close_order(target_flatten, ref_flatten)
    if close >= 3:
        return {"close_order": close}

    target_flatten[np.isnan(target_flatten)] = 0.0
    ref_flatten[np.isnan(ref_flatten)] = 0.0
    target_flatten[np.isposinf(target_flatten)] = 10000.0
    target_flatten[np.isneginf(target_flatten)] = -10000.0
    ref_flatten[np.isposinf(ref_flatten)] = 10000.0
    ref_flatten[np.isneginf(ref_flatten)] = -10000.0
    
    return {"similarity": calc_similarity_opt(target_flatten, ref_flatten)}
