from .utils import *
from .tensor_utils import *
from distutils.util import strtobool
import numpy as np
import os
from pathlib import Path
from functools import wraps

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

def check_nan_inf(tensors, names):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    if not isinstance(names, (list, tuple)):
        names = [names]
    print_log("- " + ", ".join([f"{name}: {get_tensor_info(tensor)}" for name, tensor in zip(names, tensors)]), flush=True)
    if not strtobool(os.environ.get("DBG_CHECK_NAN_INF", "0")):
        return
    pass_names = []
    fail_names = []
    for name, tensor in zip(names, tensors):
        if has_nan_inf_tensor(tensor):
            fail_names.append(name)
        else:
            pass_names.append(name)
    if pass_names:
        print_log(f"- nan and inf check: {', '.join([name for name in pass_names])} pass", flush=True)
    if fail_names:
        print_log(f"- nan and inf check: {', '.join([name for name in fail_names])} fail", flush=True)
        # pdb.set_trace()
            
def register_hook(module: torch.nn.Module):
    def pre_hook(module, args, kwargs):
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        print_log(f"{module_name} ({class_name}) forward start", flush=True)
        check_nan_inf([args, kwargs, dict(module.named_parameters())],["args", "kwargs", "parameters"])
        save_result_tensors((*args, kwargs), f"forward_input_{module_name}")
        increase_indent()
        
    def post_hook(module, input, output):
        decrease_indent()
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        check_nan_inf([output], "output")
        save_result_tensors(output, f"forward_output_{module_name}")
        print_log(f"{module_name} ({class_name}) forward end", flush=True)
        
    def pre_backward_hook(module, grad_output):
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        print_log(f"{module_name} ({class_name}) backward start", flush=True)
        check_nan_inf([grad_output, dict(module.named_parameters())], ["grad_output", "parameters"])
        save_result_tensors(grad_output, f"backward_input_{module_name}")
        increase_indent()
        
    def backward_hook(module, grad_input, grad_output):
        decrease_indent()
        module_name = TensorSaveHelper._module_names_[module]
        class_name = module._get_name()
        check_nan_inf([grad_input], "grad_input")
        save_result_tensors(grad_input, f"backward_output_{module_name}")
        print_log(f"{module_name} ({class_name}) backward end", flush=True)
    
    if not_started():
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("REAL_LOCAL_RANK", "0"))
        topic = os.environ.get("DBG_TOPIC", None)
        try:
            from deepspeed import get_accelerator
            device = get_accelerator().device_name()
        except ImportError:
            device = os.environ.get("DBG_DEVICE", "unknown")
        
        record_device(device)
        record_rank(rank)
        record_local_rank(local_rank)
        record_topic(topic)
        mark_start()
    
    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook)
    module.register_full_backward_pre_hook(pre_backward_hook)
    module.register_full_backward_hook(backward_hook)
    for name, submodule in module.named_modules(prefix="model"):
        TensorSaveHelper._module_names_[submodule] = name
    
def register_hook_for_function(func, save_name="function"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print_log(f"{save_name} {func.__name__} start", flush=True)
        check_nan_inf([args, kwargs], ["args", "kwargs"])
        save_result_tensors((*args, kwargs), f"{save_name}_{func.__name__}_input")
        increase_indent()
        res = func(*args, **kwargs)
        decrease_indent()
        check_nan_inf([res], "result")
        save_result_tensors(res, f"{save_name}_{func.__name__}_output")
        print_log(f"{save_name} {func.__name__} end", flush=True)
        return res
    return wrapper

def register_hook_for_Function(cls: torch.autograd.Function):
    cls.forward = register_hook_for_function(cls.forward, save_name=cls.__name__)
    cls.backward = register_hook_for_function(cls.backward, save_name=cls.__name__)
    return cls
    
def save_tensors(tensors, name, dir=".", save_grad_instead=False, grad_attr="grad"):
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
        np.savez(fn, **prepare_tensor_grad_for_save(tensors, grad_attr=grad_attr))
    else:
        np.savez(fn, **prepare_tensor_for_save(tensors))
    print_log(f"- saved {'grad' if save_grad_instead else 'tensor'} {name} in {fn}", flush=True)

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
   
def save_model_grads(module, step, name_filter=None, grad_attr="grad"):
    if not strtobool(os.environ.get("DBG_SAVE_ALL", "0")) and not strtobool(os.environ.get("DBG_SAVE_GRADS", "0")):
        return
    if name_filter is not None:
        save_tensors({k: v for k, v in module.named_parameters() if name_filter(k)}, f"grads_{step}", "grads", save_grad_instead=True, grad_attr=grad_attr)
    else:
        save_tensors(dict(module.named_parameters()), f"grads_{step}", "grads", save_grad_instead=True, grad_attr=grad_attr)
    # save_tensors(module.optimizer.bit16_groups, f"bit16_groups_{step}", "grads", save_grad_instead=True)

class LogReader:
    def __init__(self, dir=None, devices=["tpu", "cuda"], rank=0, topic=None):
        self.rank = rank
        self.dir = Path.cwd() if dir is None else Path(dir)
        self.topic = topic
        for device in devices:
            setattr(self.__class__, f"{device}_dir", property(lambda self, device=device: self.get_dir(device)))
            setattr(self.__class__, f"{device}_dirs", property(lambda self, device=device: self.get_dirs(device)))
    
    def get_dir(self, device):
        dirs = self.get_dirs(device, max_num=1)
        return dirs[0] if dirs else None

    def get_dirs(self, device, max_num=None):
        log_folders = list(self.dir.glob("logs_*"))
        log_folders.sort(key=lambda x: -x.stat().st_mtime)
        ret = []
        for folder in log_folders:
            if max_num is not None and len(ret) >= max_num:
                break
            log_fn = folder / f"log_{self.rank}.txt"
            if not log_fn.exists():
                continue
            log = open(log_fn, "r")
            next(log)
            if next(log).strip() == f"device: {device}, rank: {self.rank}":
                if self.topic is not None:
                    if next(log).strip() == f"Topic: {self.topic}":
                        ret.append(folder)
                        continue
                else:
                    ret.append(folder)
                    continue
        return ret