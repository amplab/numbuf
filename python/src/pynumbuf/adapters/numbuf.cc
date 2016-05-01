#include "numbuf.h"

#include "numpy.h"

namespace pynumbuf {

arrow::Status ArrowToNumBuf(std::shared_ptr<arrow::RowBatch> batch, PyObject** out) {
  PyObject* dict = PyDict_New();
  size_t size = batch->num_rows();
  auto key_array = dynamic_cast<arrow::Int8Array*>(batch->column(0).get());
  auto offset_array = dynamic_cast<arrow::Int32Array*>(batch->column(1).get());
  auto dtype_array = dynamic_cast<arrow::Int64Array*>(batch->column(2).get());
  auto dim_array = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(3)).get();
  auto dim_values = std::dynamic_pointer_cast<arrow::Int64Array>(dim_array->values());
  auto content = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(4));
  auto content_values = std::dynamic_pointer_cast<arrow::UInt8Array>(content->values());
  int32_t last_offset = 0;
  for (int i = 0; i < size; ++i) {
    int32_t offset = offset_array->Value(i);
    int32_t len = offset - last_offset;
    const char* buffer = (const char*) (key_array->raw_data() + last_offset);
    PyObject* key = PyString_FromStringAndSize(buffer, len);
    std::vector<int64_t> dims;
    for (int j = dim_array->offset(i); j < dim_array->offset(i) + dim_array->value_length(i); ++j) {
      dims.push_back(dim_values->Value(j));
    }
    const uint8_t* const_data = reinterpret_cast<const uint8_t*>(content_values->raw_data()) + content->offset(i);
    uint8_t* data = const_cast<uint8_t*>(const_data);
    PyObject* array = PyArray_SimpleNewFromData(dim_array->value_length(i), &dims[0], arrow_type_to_numpy(numbuf::arrow_type(dtype_array->Value(i))), reinterpret_cast<void*>(data));
    PyDict_SetItem(dict, key, array);

    last_offset = offset;
  }
  *out = dict;
  return arrow::Status::OK();
}

arrow::Status NumBufToArrow(PyObject* dict, std::shared_ptr<arrow::RowBatch> *out) {
  numbuf::Numbuf result;
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    if (PyString_Check(key)) {
      char* chardata;
      Py_ssize_t len;
      PyString_AsStringAndSize(key, &chardata, &len);
      if(PyArray_Check(value)) {
        PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
        size_t ndim = PyArray_NDIM(array);
        int dtype = PyArray_TYPE(array);
        std::vector<int64_t> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
          dims[i] = PyArray_DIM(array, i);
        }
        auto type = numpy_type_to_arrow(dtype);
        array = PyArray_GETCONTIGUOUS(array); // TODO(pcm): support noncontiguous arrays without copying
        auto buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<uint8_t*>(PyArray_DATA(array)), PyArray_SIZE(array) * type->value_size());
        result.add_entry(chardata, len, dims.begin(), dims.end(), type, buffer);
      } else {
        assert(false);
      }
    }
  }
  *out = result.content();
  return arrow::Status::OK();
}

}
