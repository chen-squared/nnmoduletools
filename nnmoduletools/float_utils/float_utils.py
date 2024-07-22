import warnings
import numpy as np
try:
    import torch
    use_torch = True
except ImportError:
    use_torch = False

###################
# Float Converter #
###################

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
                assert np.all(0 <= memory) and np.all(memory < (1 << self.bits))
                self.memory = np.array(memory, dtype=self.memory_dtype)

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


class FloatConverter:
    def __init__(self, value=None, memory=None):
        warnings.warn(
            "FloatConverter is deprecated. Use floating class instead.", DeprecationWarning)
        if value is None and not memory is None:
            assert isinstance(memory, (int, np.integer))
        elif not value is None and memory is None:
            assert isinstance(value, (float, np.floating, int, np.integer))
            if isinstance(value, (int, np.integer)):
                warnings.warn(
                    "Warning: You are inputting an integer as value. If it is not intended, specify 'memory=' argument.", Warning)
        else:
            raise ValueError
        self.memory = memory
        self.value = value
        self.init_memory()
        self.init_value()

    def init_memory(self):
        if not self.memory is None:
            self.fp32_memory = self.memory & 0xffffffff
            self.bfp16_memory = self.memory & 0xffff
            self.fp16_memory = self.memory & 0xffff

        if not self.value is None:
            self.fp32_memory = fp32_encode(self.value).item()
            self.bfp16_memory = bfp16_encode(self.value).item()
            self.fp16_memory = fp16_encode(self.value).item()

    def init_value(self):
        self.fp32 = fp32_decode(self.fp32_memory).item()
        self.bfp16 = bfp16_decode(self.bfp16_memory).item()
        self.fp16 = fp16_decode(self.fp16_memory).item()

    def print_results(self):
        print("BF16: %10d     0x%04x" %
              (self.bfp16_memory, self.bfp16_memory), "Value:", self.bfp16, )
        print("FP16: %10d     0x%04x" %
              (self.fp16_memory, self.fp16_memory), "Value:", self.fp16, )
        print("FP32: %10d 0x%08x" %
              (self.fp32_memory, self.fp32_memory), "Value:", self.fp32, )

