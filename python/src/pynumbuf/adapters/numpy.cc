#include "numpy.h"

#include <numbuf/tensor.h>
#include <numbuf/types.h>

namespace pynumbuf {

#define NUMPY_TYPE_TO_ARROW_CASE(TYPE)    \
  case NPY_##TYPE:                         \
    return numbuf::TYPE##_TYPE;

arrow::TypePtr numpy_type_to_arrow(int numpy_type) {
  switch (numpy_type) {
    NUMPY_TYPE_TO_ARROW_CASE(INT8)
    NUMPY_TYPE_TO_ARROW_CASE(INT16)
    NUMPY_TYPE_TO_ARROW_CASE(INT32)
    NUMPY_TYPE_TO_ARROW_CASE(INT64)
    NUMPY_TYPE_TO_ARROW_CASE(UINT8)
    NUMPY_TYPE_TO_ARROW_CASE(UINT16)
    NUMPY_TYPE_TO_ARROW_CASE(UINT32)
    NUMPY_TYPE_TO_ARROW_CASE(UINT64)
    NUMPY_TYPE_TO_ARROW_CASE(FLOAT)
    NUMPY_TYPE_TO_ARROW_CASE(DOUBLE)
    default:
      assert(false);
      return 0;  // To suppress warning that we're not returning anything
  }
}

#define ARROW_TYPE_TO_NUMPY_CASE(TYPE)    \
  case arrow::Type::TYPE:                 \
    return NPY_##TYPE;

int arrow_type_to_numpy(arrow::TypePtr type) {
  switch (type->type) {
    ARROW_TYPE_TO_NUMPY_CASE(INT8)
    ARROW_TYPE_TO_NUMPY_CASE(INT16)
    ARROW_TYPE_TO_NUMPY_CASE(INT32)
    ARROW_TYPE_TO_NUMPY_CASE(INT64)
    ARROW_TYPE_TO_NUMPY_CASE(UINT8)
    ARROW_TYPE_TO_NUMPY_CASE(UINT16)
    ARROW_TYPE_TO_NUMPY_CASE(UINT32)
    ARROW_TYPE_TO_NUMPY_CASE(UINT64)
    ARROW_TYPE_TO_NUMPY_CASE(FLOAT)
    ARROW_TYPE_TO_NUMPY_CASE(DOUBLE)
    default:
      assert(false);
      return 0;  // To suppress warning that we're not returning anything
  }
}

arrow::Status ArrowToNumPy(std::shared_ptr<arrow::RowBatch> batch, PyObject** out) {
  numbuf::Tensor tensor = numbuf::Tensor::from_arrow(batch);
  uint8_t* data = const_cast<uint8_t*>(tensor.data()); // TODO(pcm): make numpy array immutable
  std::vector<npy_intp> dims(tensor.num_dims()); // introducing temporary variable to fix compilation on MAC OS X
  for (int i = 0; i < tensor.num_dims(); ++i) {
    dims[i] = tensor.dims()[i];
  }
  *out = PyArray_SimpleNewFromData(tensor.num_dims(), dims.data(), arrow_type_to_numpy(tensor.dtype()), reinterpret_cast<void*>(data));
  return arrow::Status::OK();
}

// This assumes that array is already contiguous for now
arrow::Status NumPyToArrow(PyArrayObject* array, std::shared_ptr<arrow::RowBatch> *out) {
  size_t ndim = PyArray_NDIM(array);
  int dtype = PyArray_TYPE(array);
  std::vector<int64_t> dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    dims[i] = PyArray_DIM(array, i);
  }
  auto type = numpy_type_to_arrow(dtype);
  auto buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<uint8_t*>(PyArray_DATA(array)), PyArray_SIZE(array) * type->value_size());
  numbuf::Tensor tensor(dims.begin(), dims.end(), type, buffer);

  *out = tensor.content();

  return arrow::Status::OK();
}

}
