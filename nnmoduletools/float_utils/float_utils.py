import warnings
import numpy as np
from functools import partial
try:
    import torch
    use_torch = True
except ImportError:
    use_torch = False
    warnings.warn("PyTorch is not installed. Continuing with naive bf16 implementation and no fp8 support.", Warning)

if use_torch:
    try:
        torch.uint8
        torch.uint16
        torch.uint32
        torch.float8_e4m3fn
        torch.float8_e5m2
        all_torch = True
    except AttributeError:
        all_torch = False
        warnings.warn("PyTorch version is too old. Continuing with old implementation and no fp8 support.", Warning)

###################
# Float Converter #
###################
if use_torch and all_torch:
    from ..module_debugger.tensor_utils import apply_recursively
    
    @apply_recursively()
    def to_floating(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
            return floating(tensor, torch_dtype=tensor.dtype)
        else:
            return tensor
        
    @apply_recursively()
    def tensor_to_fp32(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
            return torch.tensor(tensor, dtype=torch.float32)
        else:
            return tensor

    class floating(torch.Tensor):
        @staticmethod
        def __new__(cls, value=None, memory=None, torch_dtype=torch.float32):
            bits = torch_dtype.itemsize * 8
            memory_dtype = eval(f"torch.uint{bits}")

            if memory is None:
                tensor = torch.as_tensor(value, dtype=torch_dtype)
            else:
                tensor = torch.as_tensor(memory, dtype=memory_dtype).view(torch_dtype)

            instance = super().__new__(cls, tensor)

            instance.bits = bits
            instance.memory_dtype = memory_dtype
            return instance
   
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            try:
                ret = super().__torch_function__(func, types, args=args, kwargs=kwargs)
            except RuntimeError as e:
                warnings.warn(
                    f"\n{e}. Casting to float32 for calculation"
                    , Warning)
                new_args = tensor_to_fp32(args)
                new_kwargs = {k: tensor_to_fp32(v) for k, v in kwargs.items()} if kwargs else {}
                target_dtype = args[0].dtype if len(args) >= 1 and isinstance(args[0], torch.Tensor) else torch.float32
                ret = func(*new_args, **new_kwargs).to(target_dtype)
            return to_floating(ret)
            
        def __repr__(self):
            dtype_name = self.dtype.__str__().split(".")[-1]
            elem_repr = f"%#.10g(0x%0{self.bits // 4}x)"
            data_repr = ", ".join([elem_repr % (v, m) for v, m in zip(self.value.flatten(), self.memory.flatten())])
            return f"{dtype_name}[{data_repr}]"

        @property
        def value(self):
            return torch.tensor(self.to(torch.float32), dtype=torch.float32)

        @property
        def memory(self):
            return torch.tensor(self.view(self.memory_dtype), dtype=self.memory_dtype)


    float32 = partial(floating, torch_dtype=torch.float32)
    fp32 = float32
    f32 = float32
    float16 = partial(floating, torch_dtype=torch.float16)
    fp16 = float16
    f16 = float16
    bfloat16 = partial(floating, torch_dtype=torch.bfloat16)
    bf16 = bfloat16
    bfp16 = bfloat16
    float8_e4m3 = partial(floating, torch_dtype=torch.float8_e4m3fn)
    fp8_e4m3 = float8_e4m3
    float8_e5m2 = partial(floating, torch_dtype=torch.float8_e5m2)
    fp8_e5m2 = float8_e5m2
else:
    # Regulations:
    # memory: np.uint16 or np.uint32
    # value: all convert to np.float32

    def fp32_decode(fp32_memory):
        return np.array(fp32_memory, dtype=np.uint32).view(np.float32)

    def fp32_encode(fp32_value):
        return np.array(fp32_value, dtype=np.float32).view(np.uint32)

    def fp16_decode(fp16_memory):
        return np.array(fp16_memory, dtype=np.uint16).view(np.float16).astype(np.float32)

    def fp16_encode(fp16_value):
        return np.array(fp16_value, dtype=np.float16).view(np.uint16)

    def bfp16_decode(bfp16_memory):
        if use_torch:
            bfp16_memory = np.array(bfp16_memory, dtype=np.uint16).view(np.int16)
            return torch.tensor(bfp16_memory, dtype=torch.int16).view(torch.bfloat16).to(torch.float32).numpy()
        else:
            bfp16_memory = np.array(bfp16_memory, dtype=np.uint32) << 16
            return np.array(bfp16_memory, dtype=np.uint32).view(np.float32)

    def bfp16_encode(bfp16_value):
        if use_torch:
            return torch.tensor(bfp16_value, dtype=torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        else:
            bfp16_memory = np.array(bfp16_value, dtype=np.float32).view(np.uint32) 
            mask = (bfp16_memory & 0x1FFFF) == 0x8000
            bfp16_memory[mask] -= 0x8000
            bfp16_memory += 0x8000
            return (bfp16_memory >> 16).astype(np.uint16)


    class floating:
        bits = 32
        memory_dtype = np.uint32

        def __init__(self, value=None, memory=None):
            if memory is None:
                assert not value is None
                if isinstance(value, floating):
                    self.memory = self.encode(value.value)
                else:
                    self.memory = self.encode(value)
            else:
                assert value is None
                if isinstance(memory, floating):
                    self.memory = self.encode(memory.value)
                else:
                    memory = np.array(memory, dtype=self.memory_dtype)
                    assert np.all(0 <= memory) and np.all(memory < (1 << self.bits))
                    self.memory = memory

        def __repr__(self):
            elem_repr = f"%#.10g(0x%0{self.bits // 4}x)"
            data_repr = ", ".join([elem_repr % (v, m) for v, m in zip(self.value.flatten(), self.memory.flatten())])
            return f"{self.__class__.__name__}[{data_repr}]"

        def __str__(self):
            return str(self.value)

        def __int__(self):
            return int(self.value)

        def __abs__(self):
            return self.__class__(abs(self.value))

        def __eq__(self, other):
            return self.value == self.__class__(other).value

        def __lt__(self, other):
            return self.value < self.__class__(other).value

        def __add__(self, other):
            return self.__class__(self.value + self.__class__(other).value)

        def __radd__(self, other):
            return self.__class__(self.__class__(other).value + self.value)

        def __sub__(self, other):
            return self.__class__(self.value - self.__class__(other).value)

        def __rsub__(self, other):
            return self.__class__(self.__class__(other).value - self.value)

        def __mul__(self, other):
            return self.__class__(self.value * self.__class__(other).value)

        def __rmul__(self, other):
            return self.__class__(self.__class__(other).value * self.value)

        def __truediv__(self, other):
            return self.__class__(self.value / self.__class__(other).value)

        def __rtruediv__(self, other):
            return self.__class__(self.__class__(other).value / self.value)

        def __pos__(self):
            return self.__class__(self.value)

        def __neg__(self):
            return self.__class__(-self.value)

        @staticmethod
        def decode(memory):
            return fp32_decode(memory)

        @staticmethod
        def encode(flt):
            return fp32_encode(flt)

        @property
        def value(self):
            return self.decode(self.memory)
        
        def to(self, dtype):
            if issubclass(dtype, floating):
                return dtype(self.value)
            else:
                raise ValueError


    f32 = floating
    fp32 = floating
    float32 = floating


    class float16(floating):
        bits = 16
        memory_dtype = np.uint16

        def __init__(self, value=None, memory=None):
            super().__init__(value, memory)

        @staticmethod
        def decode(memory):
            return fp16_decode(memory)

        @staticmethod
        def encode(flt):
            return fp16_encode(flt)


    f16 = float16
    fp16 = float16


    class bfloat16(floating):
        bits = 16
        memory_dtype = np.uint16

        def __init__(self, value=None, memory=None):
            super().__init__(value, memory)

        @staticmethod
        def decode(memory):
            return bfp16_decode(memory)

        @staticmethod
        def encode(flt):
            return bfp16_encode(flt)


    bf16 = bfloat16
    bfp16 = bfloat16
    
    float8_e4m3 = NotImplemented
    fp8_e4m3 = float8_e4m3
    float8_e5m2 = NotImplemented
    fp8_e5m2 = float8_e5m2
