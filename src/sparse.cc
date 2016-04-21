#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL ETHER_ARRAY_API

#include "api.h"
#include "array.h"

#include <iostream>

std::shared_ptr<arrow::Schema> csr_sparse_header_schema() {
  auto info_field = std::make_shared<arrow::Field>("info", std::make_shared<arrow::Int64Type>());
	return std::shared_ptr<arrow::Schema>(new arrow::Schema({info_field}));
}

std::shared_ptr<arrow::RowBatch> make_csr_matrix_header(PyObject* matrix, int64_t data_offset, arrow::MemoryPool* pool) {
  PyObject* shape = PyObject_GetAttr(matrix, PyString_FromString("shape")); // TODO: this might leak memory of the string
  std::vector<int64_t>* header_data = new std::vector<int64_t>({PyInt_AsLong(PyTuple_GetItem(shape, 0)), PyInt_AsLong(PyTuple_GetItem(shape, 1)), data_offset}); // TODO: deallocate
  std::shared_ptr<arrow::Buffer> header_buffer = to_buffer(*header_data);
  auto header_array = std::make_shared<arrow::Int64Array>(header_data->size(), header_buffer);
	auto schema = csr_sparse_header_schema();
  return std::shared_ptr<arrow::RowBatch>(new arrow::RowBatch(schema, (int)header_data->size(), {header_array}));
}

std::shared_ptr<arrow::Schema> csr_sparse_schema() {
  auto indices_field = std::make_shared<arrow::Field>("indices", std::make_shared<arrow::Int32Type>());
  auto indptr_field = std::make_shared<arrow::Field>("indptr", std::make_shared<arrow::Int32Type>());
  auto data_field = std::make_shared<arrow::Field>("data", std::make_shared<arrow::DoubleType>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({indices_field, indptr_field, data_field}));
}

#define SERIALIZE_FLAT_ARRAY(TYPE) \
  case TYPE: \
    data = std::make_shared<arrow::Buffer>(reinterpret_cast<uint8_t*>(PyArray_DATA(array)), sizeof(npy_traits<TYPE>::value_type) * size); \
    return std::make_shared<typename npy_traits<TYPE>::ArrayType>(size, data);

std::shared_ptr<arrow::Array> serialize_flat_array(PyArrayObject* array) {
  assert(PyArray_NDIM(array) == 1);
  npy_intp size = PyArray_SIZE(array);
  std::shared_ptr<arrow::Buffer> data;
  switch (PyArray_TYPE(array)) {
    SERIALIZE_FLAT_ARRAY(NPY_INT8)
    SERIALIZE_FLAT_ARRAY(NPY_INT16)
    SERIALIZE_FLAT_ARRAY(NPY_INT32)
    SERIALIZE_FLAT_ARRAY(NPY_INT64)
    SERIALIZE_FLAT_ARRAY(NPY_UINT8)
    SERIALIZE_FLAT_ARRAY(NPY_UINT16)
    SERIALIZE_FLAT_ARRAY(NPY_UINT32)
    SERIALIZE_FLAT_ARRAY(NPY_UINT64)
    SERIALIZE_FLAT_ARRAY(NPY_FLOAT)
    SERIALIZE_FLAT_ARRAY(NPY_DOUBLE)
    default:
      std::cout << "serialize_flat_array error: type not known" << std::endl;
      exit(1);
  }
}

// TODO: figure out how to make an immutable array
#define DESERIALIZE_FLAT_ARRAY(TYPE) \
  case TYPE: { \
    auto primitive_array = dynamic_cast<npy_traits<TYPE>::ArrayType*>(array.get()); \
    return PyArray_SimpleNewFromData(dims.size(), &dims[0], TYPE, (void*)primitive_array->raw_data()); \
  }

// TODO: this only supports double at this point
PyObject* deserialize_flat_array(int64_t npy_type, std::shared_ptr<arrow::Array> array) {
  std::vector<npy_intp> dims;
  dims.push_back(array->length());
  switch (npy_type) {
    DESERIALIZE_FLAT_ARRAY(NPY_INT8)
    DESERIALIZE_FLAT_ARRAY(NPY_INT16)
    DESERIALIZE_FLAT_ARRAY(NPY_INT32)
    DESERIALIZE_FLAT_ARRAY(NPY_INT64)
    DESERIALIZE_FLAT_ARRAY(NPY_UINT8)
    DESERIALIZE_FLAT_ARRAY(NPY_UINT16)
    DESERIALIZE_FLAT_ARRAY(NPY_UINT32)
    DESERIALIZE_FLAT_ARRAY(NPY_UINT64)
    DESERIALIZE_FLAT_ARRAY(NPY_FLOAT)
    DESERIALIZE_FLAT_ARRAY(NPY_DOUBLE)
    default:
      std::cout << "deserialize_flat_array error: type not known" << std::endl;
      exit(1);
  }
}

bool is_csr_matrix(PyObject* matrix) {
  PyObject* module = PyImport_ImportModule("scipy.sparse");
  PyObject* module_dict = PyModule_GetDict(module);
  PyObject* protocol_class = PyDict_GetItemString(module_dict, "csr_matrix");
  return PyObject_IsInstance(matrix, protocol_class);
}

std::shared_ptr<arrow::RowBatch> serialize_csr(PyObject* matrix, arrow::MemoryPool* pool) {
	PyObject* shape = PyObject_GetAttr(matrix, PyString_FromString("shape")); // TODO: this might leak memory of the string
	int nnz = PyInt_AsLong(PyObject_GetAttr(matrix, PyString_FromString("nnz"))); // TODO: this might leak memory of the string
	PyObject *indices = PyObject_GetAttr(matrix, PyString_FromString("indices")); // TODO: this might leak memory of the string
	assert(PyArray_Check(indices));
	auto indices_array = serialize_flat_array((PyArrayObject*) indices);
	PyObject *indptr = PyObject_GetAttr(matrix, PyString_FromString("indptr")); // TODO: this might leak memory of the string
	assert(PyArray_Check(indptr));
	auto indptr_array = serialize_flat_array((PyArrayObject*) indptr);
	PyObject *data = PyObject_GetAttr(matrix, PyString_FromString("data")); // TODO: this might leak memory of the string
	assert(PyArray_Check(data));
	auto data_array = serialize_flat_array((PyArrayObject*) data);
	auto schema = csr_sparse_schema();
	return std::shared_ptr<arrow::RowBatch>(new arrow::RowBatch(schema, nnz, {indices_array, indptr_array, data_array}));
}

PyObject* deserialize_csr(std::shared_ptr<arrow::RowBatch> matrix, int64_t num_rows, int64_t num_cols) {
  PyObject* indices = deserialize_flat_array(NPY_INT32, matrix->column(0));
	PyObject* indptr = deserialize_flat_array(NPY_INT32, matrix->column(1));
	PyObject* data = deserialize_flat_array(NPY_DOUBLE, matrix->column(2));
	return PyTuple_Pack(5, PyInt_FromLong(num_rows), PyInt_FromLong(num_cols), indices, indptr, data);
}
