import os
import time
import glob
import warnings

class GlobalVar:
    _print_indent_ = 0
    _start_time_stamp_ = None
    _log_dir_ = None
    _rank_ = None
    _local_rank_ = None
    _device_ = None
    _topic_ = None


def get_indent():
    return '|   ' * GlobalVar._print_indent_

def increase_indent():
    GlobalVar._print_indent_ += 1
    
def decrease_indent():
    GlobalVar._print_indent_ -= 1
    
def print_log(msg, flush=False, end='\n'):
    print(f"{get_indent()}{msg}", end=end, flush=flush)
    if not not_started():
        with open(os.path.join(GlobalVar._log_dir_, f"log_{read_rank()}.txt"), 'a') as f:
            f.write(f"{get_indent()}{msg}{end}")
            f.flush()
    else:
        warnings.warn("Log file not specified. Logs are only printed to screen.")
    
def mark_start():
    if not_started():
        local_rank = read_local_rank()
        if local_rank == 0:
            GlobalVar._start_time_stamp_ = time.strftime("%Y.%m.%d.%H.%M.%S")
            set_log_dir(os.path.join(os.getcwd(), f"logs_{read_start_time_stamp()}"))
            os.makedirs(get_log_dir(), exist_ok=True)
        else:
            cnt = 0
            while 1:
                log_folders = glob.glob(os.path.join(os.getcwd(), "logs_*"))
                log_folders.sort(key=lambda x: os.path.getmtime(x))
                if len(log_folders) == 0:
                    time.sleep(1)
                    continue
                closest_folder = log_folders[-1]
                
                current_time = time.time()
                closest_time_fn = os.path.basename(closest_folder).replace("logs_", "")
                try:
                    closest_time = time.mktime(time.strptime(closest_time_fn, "%Y.%m.%d.%H.%M.%S"))
                except ValueError:
                    continue
                if 0 <= current_time - closest_time < 10:
                    break
                else:
                    time.sleep(1)
                    cnt += 1
                    if cnt > 10:
                        raise ValueError("Cannot find the start time stamp")

            set_log_dir(closest_folder)
            GlobalVar._start_time_stamp_ = closest_time_fn

    print_log(f"Start time: {read_start_time_stamp()}")
    print_log(f"device: {read_device()}, rank: {read_rank()}")
    print_log(f"Topic: {read_topic()}")

def not_started():
    return GlobalVar._start_time_stamp_ is None

def read_start_time_stamp():
    assert GlobalVar._start_time_stamp_ is not None
    return GlobalVar._start_time_stamp_

def set_log_dir(log_dir):
    assert GlobalVar._log_dir_ is None
    GlobalVar._log_dir_ = log_dir

def get_log_dir():
    assert GlobalVar._log_dir_ is not None
    return GlobalVar._log_dir_

def get_log_file_path(fn):
    new_fn = os.path.join(get_log_dir(), fn)
    os.makedirs(os.path.dirname(new_fn), exist_ok=True)
    return new_fn

def record_rank(rank):
    assert GlobalVar._rank_ is None
    GlobalVar._rank_ = rank
  
def read_rank():
    assert GlobalVar._rank_ is not None
    return GlobalVar._rank_

def record_local_rank(rank):
    assert GlobalVar._local_rank_ is None
    GlobalVar._local_rank_ = rank
  
def read_local_rank():
    assert GlobalVar._local_rank_ is not None
    return GlobalVar._local_rank_

def record_device(device):
    assert GlobalVar._device_ is None
    GlobalVar._device_ = device
    
def read_device():
    assert GlobalVar._device_ is not None
    return GlobalVar._device_

def record_topic(topic):
    assert GlobalVar._topic_ is None
    GlobalVar._topic_ = topic
    
def read_topic():
    return GlobalVar._topic_