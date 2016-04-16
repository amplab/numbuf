#include "api.h"

std::shared_ptr<arrow::RowBatch> make_array_header(PyArrayObject* array, int64_t data_offset, arrow::MemoryPool* pool) {
	/*
  size_t ndim = PyArray_NDIM(array);
  auto array_header_builder = make_int64_builder(pool);
	array_header_builder->Append(PyArray_TYPE(array));
  for (size_t i = 0; i < ndim; ++i) {
    array_header_builder->Append(PyArray_DIM(array, i));
  }
	return array_header_builder->Finish();
	*/
}

std::shared_ptr<arrow::RowBatch> serialize_array(PyArrayObject* array, arrow::MemoryPool* pool) {

}

PyObject* deserialize_array(std::shared_ptr<arrow::RowBatch> array_header, std::shared_ptr<arrow::RowBatch> array) {

}
