#include "python.h"

arrow::Status ArrowToPyDict(std::shared_ptr<arrow::RowBatch> batch, PyObject** out) {
  PyObject* dict = PyDict_New();
  size_t size = batch->num_rows();
  auto key_array = dynamic_cast<arrow::Int8Array*>(batch->column(0).get());
  auto offset_array = dynamic_cast<arrow::Int32Array*>(batch->column(1).get());
  auto val_array = dynamic_cast<arrow::Int64Array*>(batch->column(2).get());
  int32_t last_offset = 0;
  for (int i = 0; i < size; ++i) {
    int32_t offset = offset_array->Value(i);
    int32_t len = offset - last_offset;
    const char* buffer = (const char*) (key_array->raw_data() + last_offset);
    PyObject* key = PyString_FromStringAndSize(buffer, len);
    PyDict_SetItem(dict, key, PyInt_FromLong(val_array->Value(i)));
    last_offset = offset;
  }
  *out = dict;
  return arrow::Status::OK();
}

arrow::Status PyDictToArrow(PyObject* dict, std::shared_ptr<arrow::RowBatch> *out) {
  numbuf::Dict result;
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    if (PyString_Check(key)) {
      char* buffer;
      Py_ssize_t len;
      PyString_AsStringAndSize(key, &buffer, &len);
      result.add_entry(buffer, len, PyInt_AsLong(value));
    }
  }
  *out = result.content();
  return arrow::Status::OK();
}
