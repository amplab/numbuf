#ifndef ETHER_ARRAY_H
#define ETHER_ARRAY_H

template <int TYPE>
struct npy_traits {
};

template <>
struct npy_traits<NPY_BOOL> {
  typedef uint8_t value_type;
  static const std::shared_ptr<arrow::BooleanType> primitive_type;
  using ArrayType = arrow::BooleanArray;
};

const std::shared_ptr<arrow::BooleanType> npy_traits<NPY_BOOL>::primitive_type = std::make_shared<arrow::BooleanType>();

#define NPY_INT_DECL(TYPE, CapType, T)                                 \
  template <>                                                          \
  struct npy_traits<NPY_##TYPE> {                                      \
    typedef T value_type;                                              \
    static const std::shared_ptr<arrow::CapType##Type> primitive_type; \
    using ArrayType = arrow::CapType##Array;                           \
  };                                                                   \
                                                                       \
  const std::shared_ptr<arrow::CapType##Type> npy_traits<NPY_##TYPE>::primitive_type = std::make_shared<arrow::CapType##Type>();

NPY_INT_DECL(INT8, Int8, int8_t);
NPY_INT_DECL(INT16, Int16, int16_t);
NPY_INT_DECL(INT32, Int32, int32_t);
NPY_INT_DECL(INT64, Int64, int64_t);
NPY_INT_DECL(UINT8, UInt8, uint8_t);
NPY_INT_DECL(UINT16, UInt16, uint16_t);
NPY_INT_DECL(UINT32, UInt32, uint32_t);
NPY_INT_DECL(UINT64, UInt64, uint64_t);

template <>
struct npy_traits<NPY_FLOAT32> {
  typedef float value_type;
  static const std::shared_ptr<arrow::FloatType> primitive_type;
  using ArrayType = arrow::FloatArray;
};

const std::shared_ptr<arrow::FloatType> npy_traits<NPY_FLOAT32>::primitive_type = std::make_shared<arrow::FloatType>();

template <>
struct npy_traits<NPY_FLOAT64> {
  typedef double value_type;
  static const std::shared_ptr<arrow::DoubleType> primitive_type;
  using ArrayType = arrow::DoubleArray;
};

const std::shared_ptr<arrow::DoubleType> npy_traits<NPY_FLOAT64>::primitive_type = std::make_shared<arrow::DoubleType>();

template <>
struct npy_traits<NPY_OBJECT> {
  typedef PyObject* value_type;
};

#endif
