import struct 
from enum import Enum

WEIGHTS_FILE = "/home/dhaya/projects/mytinygrad/tinygrad/weights/gemma-2b.gguf"

class ValueType(Enum):
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    STRING = 8
    ARRAY = 9
    UINT64 = 10

def read_string(f):
    len = struct.unpack("<Q", f.read(8))[0]
    return f.read(len).decode("utf-8")

def read_value(f, val_type):    
    if val_type == ValueType.UINT32.value:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == ValueType.INT32.value:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == ValueType.FLOAT32.value:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == ValueType.STRING.value:
        return read_string(f)
    elif val_type == ValueType.UINT64.value:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == ValueType.ARRAY.value:
        data_type, count = struct.unpack("<IQ", f.read(4+8))
        return [read_value(f, data_type) for _ in range(count)]        
    else:
        raise NotImplementedError(f"Data type {val_type} not implemented")

# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
# https://github.com/99991/pygguf/blob/main/gguf.py
def load_gguf(weights):    
    with open(weights, "rb") as f:
        f.seek(0)
        assert f.read(4) == b"GGUF"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(4+8+8))
        assert version == 3        
        print(version, n_tensors, n_kv)

        info = {}
        for _ in range(n_kv):
            name = read_string(f)
            val_type = read_value(f, ValueType.UINT32.value)
            val = read_value(f, val_type)
            print(f'{name=},{val_type=}')            
            if val_type != ValueType.ARRAY.value:
                print(f'{val=}')

        tensorinfo = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = read_value(f, ValueType.UINT32.value)
            shape = [read_value(f, ValueType.UINT64.value) for _ in range(n_dims)]
            ggml_type = read_value(f, ValueType.UINT32.value)
            offset = read_value(f, ValueType.UINT64.value)

            tensorinfo[name] = {
                "ggml_type": ggml_type,
                "shape": shape,
                "offset": offset,
            }
        # current position
        start = f.tell()
        print(tensorinfo)
    
        # TODO: figure out actual offsets
        # return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;


if __name__ == "__main__":
    print("hello world")
    load_gguf(WEIGHTS_FILE)


