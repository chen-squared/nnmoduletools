import struct
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


def fp32_decode(fp32):
    if use_torch:
        if fp32 >> 31:
            fp32 -= 0x100000000
        return torch.tensor(fp32, dtype=torch.int32).view(torch.float32).item()
    return struct.unpack('!f', struct.pack('!I', fp32))[0]


def fp32_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.float32).view(torch.int32).item()
        if memory < 0:
            memory += 0x100000000
        return memory
    return struct.unpack('!I', struct.pack('!f', flt))[0]


def fp16_decode(fp16):
    if use_torch:
        if fp16 >> 15:
            fp16 -= 0x10000
        return torch.tensor(fp16, dtype=torch.int16).view(torch.float16).item()
    return struct.unpack('<e', struct.pack('<H', fp16))[0]


def fp16_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.float16).view(torch.int16).item()
        if memory < 0:
            memory += 0x10000
        return memory
    return struct.unpack('<H', np.float16(flt))[0]


def bfp16_decode(bfp16):
    if use_torch:
        if bfp16 >> 15:
            bfp16 -= 0x10000
        return torch.tensor(bfp16, dtype=torch.int16).view(torch.bfloat16).item()
    return struct.unpack('!f', struct.pack('!I', bfp16 << 16))[0]


def bfp16_encode(flt):
    if use_torch:
        memory = torch.tensor(flt, dtype=torch.bfloat16).view(torch.int16).item()
        if memory < 0:
            memory += 0x10000
        return memory
    i = struct.unpack('!I', struct.pack('!f', flt))[0]
    if (i & 0x1FFFF) == 0x8000:
        i -= 0x8000
    i += 0x8000
    return i >> 16


class floating:
    bits = 32

    def __init__(self, value=None, memory=None):
        if memory is None:
            assert not value is None
            self.value = self.decode(self.encode(value))
        else:
            assert value is None
            assert 0 <= memory < (1 << self.bits)
            self.value = self.decode(memory)

    def __repr__(self):
        return f"%s(%s)[0x%0{self.bits // 4}x]" % (self.__class__.__name__, self.value, self.memory)

    def __str__(self):
        return str(float(self))

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __abs__(self):
        return self.__class__(abs(self.value))

    def __eq__(self, other):
        return self.value == float(other)

    def __lt__(self, other):
        return self.value < float(other)

    def __add__(self, other):
        return self.__class__(self.value + float(other))

    def __radd__(self, other):
        return self.__class__(float(other) + self.value)

    def __sub__(self, other):
        return self.__class__(self.value - float(other))

    def __rsub__(self, other):
        return self.__class__(float(other) - self.value)

    def __mul__(self, other):
        return self.__class__(self.value * float(other))

    def __rmul__(self, other):
        return self.__class__(float(other) * self.value)

    def __truediv__(self, other):
        return self.__class__(self.value / float(other))

    def __rtruediv__(self, other):
        return self.__class__(float(other) / self.value)

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
    def memory(self):
        return self.encode(self.value)


f32 = floating
fp32 = floating


class float16(floating):
    bits = 16

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
        if value is None and not memory is None:
            assert isinstance(memory, (int, np.integer))
        elif not value is None and memory is None:
            assert isinstance(value, (float, np.floating, int, np.integer))
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
            self.fp32_memory = fp32_encode(self.value)
            self.bfp16_memory = bfp16_encode(self.value)
            self.fp16_memory = fp16_encode(self.value)

    def init_value(self):
        self.fp32 = fp32_decode(self.fp32_memory)
        self.bfp16 = bfp16_decode(self.bfp16_memory)
        self.fp16 = fp16_decode(self.fp16_memory)

    def print_results(self):
        print("BF16: %10d     0x%04x" %
              (self.bfp16_memory, self.bfp16_memory), "Value:", self.bfp16, )
        print("FP16: %10d     0x%04x" %
              (self.fp16_memory, self.fp16_memory), "Value:", self.fp16, )
        print("FP32: %10d 0x%08x" %
              (self.fp32_memory, self.fp32_memory), "Value:", self.fp32, )

