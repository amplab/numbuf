#include "numbuf.h"

#include "numpy.h"

namespace pynumbuf {

arrow::Status ArrowToNumBuf(std::shared_ptr<arrow::RowBatch> batch, PyObject** out) {

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
