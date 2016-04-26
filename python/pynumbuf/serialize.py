import numpy as np

def serializable(value):
  return type(value) == np.ndarray and value.dtype.name in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]
