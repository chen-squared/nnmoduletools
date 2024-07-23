from .utils import *
from .tensor_utils import *
from distutils.util import strtobool
import numpy as np
import os
from pathlib import Path

class TensorSaveHelper:
    _module_names_ = {}
    _tensor_names_ = {}
    _input_list = []
    _output_list = []
    _result_list = []

def get_tensor_name(name):
    count = TensorSaveHelper._tensor_names_.get(name, 0)
    TensorSaveHelper._tensor_names_[name] = count + 1
    return name if count == 0 else f"{name}_{count}"

def cache_result(name):
    if strtobool(os.environ.get("DBG_SAVE_IO_SEPERATELY", "0")):
        if "ward_input_" in name:
            TensorSaveHelper._input_list.append(name)
        elif "ward_output_" in name:
            TensorSaveHelper._output_list.append(name)
    else:
        TensorSaveHelper._result_list.append(name)

def combine_npz(step):
    for io, lst in (("input", TensorSaveHelper._input_list), ("output", TensorSaveHelper._output_list), ("result", TensorSaveHelper._result_list)):
        to_save = {}
        for name in lst:
            to_save[name] = dict(np.load(get_log_file_path(f"results/rank_{read_rank()}_{name}.npz")).items())
            os.remove(get_log_file_path(f"results/rank_{read_rank()}_{name}.npz"))
        if to_save:
            save_fn = get_log_file_path(f"results/rank_{read_rank()}_{io}_{step}.npz")
            np.savez(save_fn, **to_flat_dict(to_save))
            print_log(f"- saved combined {io} in {save_fn}", flush=True)
            del to_save
    TensorSaveHelper._input_list = []
    TensorSaveHelper._output_list = []
    TensorSaveHelper._result_list = []
    TensorSaveHelper._tensor_names_ = {}

def check_nan_inf(tensor, name):
    if has_nan_inf_tensor(tensor):
        print_log(f"- nan or inf in {name}", flush=True)
        # pdb.set_trace()
            
def register_hook(module: torch.nn.Module):
    def pre_hook(module, args, kwargs):
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        print_log(f"{module_name} ({class_name}) forward start", flush=True)
        save_result_tensors((*args, kwargs), f"forward_input_{module_name}")
        check_nan_inf(args, "args")
        check_nan_inf(kwargs, "kwargs")
        check_nan_inf(module.parameters(), "parameters")
        increase_indent()
        
    def post_hook(module, input, output):
        decrease_indent()
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        save_result_tensors(output, f"forward_output_{module_name}")
        check_nan_inf(output, "output")
        print_log(f"{module_name} ({class_name}) forward end", flush=True)
        
    def pre_backward_hook(module, grad_output):
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        print_log(f"{module_name} ({class_name}) backward start", flush=True)
        save_result_tensors(grad_output, f"backward_input_{module_name}")
        check_nan_inf(grad_output, "grad_output")
        check_nan_inf(module.parameters(), "parameters")
        increase_indent()
        
    def backward_hook(module, grad_input, grad_output):
        decrease_indent()
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        save_result_tensors(grad_input, f"backward_output_{module_name}")
        check_nan_inf(grad_input, "grad_input")
        print_log(f"{module_name} ({class_name}) backward end", flush=True)
    
    if not_started():
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("REAL_LOCAL_RANK", "0"))
        try:
            from deepspeed import get_accelerator
            device = get_accelerator().device_name()
        except ImportError:
            device = os.environ.get("DBG_DEVICE", "unknown")

        record_device(device)
        record_rank(rank)
        record_local_rank(local_rank)
        mark_start()
    
    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook)
    module.register_full_backward_pre_hook(pre_backward_hook)
    module.register_full_backward_hook(backward_hook)
    for name, submodule in module.named_modules(prefix="model"):
        TensorSaveHelper._module_names_[submodule] = name
    
def save_tensors(tensors, name, dir=".", save_grad_instead=False):
    if not strtobool(os.environ.get("DBG_SAVE_ALL", "0")):
        return
    fn = get_log_file_path(f"{dir}/rank_{read_rank()}_{name}.npz")
    cnt = 0
    while os.path.exists(fn):
        cnt += 1
        if cnt == 1:
            print_log(f"- Warning: {fn} already exists", flush=True)
        fn2 = get_log_file_path(f"{dir}/rank_{read_rank()}_{name}_{cnt}.npz")
        fn = fn2
    if cnt > 0: print_log(f"- renaming to{fn}", flush=True)
    if save_grad_instead:
        np.savez(fn, **prepare_tensor_grad_for_save(tensors))
    else:
        np.savez(fn, **prepare_tensor_for_save(tensors))
    print_log(f"- saved {'grad' if save_grad_instead else 'tensor'} {name.replace('_', ' ')} in {fn}", flush=True)

def save_result_tensors(tensors, name):
    white_list = os.environ.get("DBG_SAVE_WHITELIST", "")
    black_list = os.environ.get("DBG_SAVE_BLACKLIST", "")
    if not strtobool(os.environ.get("DBG_SAVE_ALL", "0")) and not strtobool(os.environ.get("DBG_SAVE_RESULTS", "0")) \
       and not white_list and not black_list:
        return
    if white_list:
        if not any([w in name for w in white_list.split(",")]):
            return
    if black_list:
        if any([b in name for b in black_list.split(",")]):
            return
    ts_name = get_tensor_name(name)
    cache_result(ts_name)
    save_tensors(tensors, ts_name, "results")

def save_model_params(module, step, name_filter=None):
    if not strtobool(os.environ.get("DBG_SAVE_ALL", "0")) and not strtobool(os.environ.get("DBG_SAVE_PARAMS", "0")):
        return
    if name_filter is not None:
        save_tensors({k: v for k, v in module.named_parameters() if name_filter(k)}, f"params_{step}", "params")
    else:
        save_tensors(dict(module.named_parameters()), f"params_{step}", "params")
   
def save_model_grads(module, step, name_filter=None):
    if not strtobool(os.environ.get("DBG_SAVE_ALL", "0")) and not strtobool(os.environ.get("DBG_SAVE_GRADS", "0")):
        return
    if name_filter is not None:
        save_tensors({k: v for k, v in module.named_parameters() if name_filter(k)}, f"grads_{step}", "grads", save_grad_instead=True)
    else:
        save_tensors(dict(module.named_parameters()), f"grads_{step}", "grads", save_grad_instead=True)
    # save_tensors(module.optimizer.bit16_groups, f"bit16_groups_{step}", "grads", save_grad_instead=True)

class LogReader:
    def __init__(self, dir=None, devices=["tpu", "cuda"], rank=0):
        self.rank = rank
        self.dir = os.getcwd() if dir is None else dir
        for device in devices:
            setattr(self.__class__, f"{device}_dir", property(lambda self, device=device: self.get_dir(device)))
    
    def get_dir(self, device):
        log_folders = glob.glob(os.path.join(self.dir, f"logs_*"))
        log_folders.sort(key=lambda x: -os.path.getmtime(x))
        for folder in log_folders:
            log = open(Path(folder) / f"log_{self.rank}.txt", "r")
            next(log)
            if device in next(log):
                return Path(folder)