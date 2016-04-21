#ifndef ETHER_API_H
#define ETHER_API_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <arrow/api.h>
#include <arrow/ipc/memory.h>
#include <arrow/ipc/adapter.h>
#include <vector>

enum EtherType {
	ARRAY_TYPE = 0,
	DICT_TYPE = 1,
	CSR_MATRIX = 2
};

/*
class Tensor {
  Tensor(std::initializer_list<int64_t> dims);
  std::shared_ptr<arrow::Schema> header_schema(); // eventually this will be schema() and header as well as data will be collapsed into a struct
  std::shared_ptr<arrow::RowBatch> header_content();
  std::shared_ptr<arrow::Schema> data_schema();
  std::shared_ptr<arrow::RowBatch> data_content();
}
*/

std::shared_ptr<arrow::StringBuilder> make_string_builder(arrow::MemoryPool* pool);
std::shared_ptr<arrow::Int64Builder> make_int64_builder(arrow::MemoryPool* pool);

std::shared_ptr<arrow::Schema> dict_schema();
std::shared_ptr<arrow::RowBatch> serialize_dict(PyObject* dict, arrow::MemoryPool* pool);
PyObject* deserialize_dict(std::shared_ptr<arrow::RowBatch> rows);

std::shared_ptr<arrow::Schema> array_header_schema();
std::shared_ptr<arrow::Schema> array_schema();
std::shared_ptr<arrow::RowBatch> make_array_header(PyArrayObject* array, int64_t data_offset, arrow::MemoryPool* pool);
std::shared_ptr<arrow::RowBatch> serialize_array(PyArrayObject* array, arrow::MemoryPool* pool);
PyObject* deserialize_array(std::shared_ptr<arrow::RowBatch> array);

std::shared_ptr<arrow::Schema> csr_sparse_header_schema();
std::shared_ptr<arrow::Schema> csr_sparse_schema();
bool is_csr_matrix(PyObject* matrix);
std::shared_ptr<arrow::RowBatch> make_csr_matrix_header(PyObject* matrix, int64_t data_offset, arrow::MemoryPool* pool);
std::shared_ptr<arrow::RowBatch> serialize_csr(PyObject* matrix, arrow::MemoryPool* pool);
PyObject* deserialize_csr(std::shared_ptr<arrow::RowBatch> matrix, int64_t num_rows, int64_t num_cols);

std::shared_ptr<arrow::Schema> make_header_schema();
std::shared_ptr<arrow::RowBatch> make_header(EtherType type, int64_t metadata_offset, arrow::MemoryPool* pool);

// ArrowObject serialize_object(PyObject* python_object);
// PyObject* deserialize_object(const ArrowObject& arrow_object);

// void assemble_object(PyObject* python_object, MemoryPool* pool, void** data_ptr, int64_t* size, int64_t* metadata_offset);
// PyObject disassemble_object(void* data_ptr, int64_t size);

template <typename T>
std::shared_ptr<arrow::Buffer> to_buffer(const std::vector<T>& values) {
  return std::make_shared<arrow::Buffer>(
      reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

// TODO: Think about error handling here

class PyObjectWriter {
public:
  PyObjectWriter(PyObject* python_object, arrow::MemoryPool* pool);
  int64_t assemble_payload_and_return_size();
  int64_t write_object_and_return_metadata_offset(arrow::ipc::MemorySource* target);
// private:
  PyObject* python_object_;
  arrow::MemoryPool* pool_;
  std::shared_ptr<arrow::RowBatch> data_payload_;
  int64_t data_payload_size_;
  int64_t data_header_size_;
  int64_t metadata_size_;
};

PyObject* read_arrow_object(arrow::ipc::MemorySource* source, int64_t metadata_offset);
// void read_arrow_object(arrow::ipc::MemorySource* source, int64_t metadata_offset);
// void write_python_object(PyObject* python_object, MemoryPool* pool, arrow::ipc::MemorySource* target, int64_t* metadata_offset);

class MemoryMapFixture {
 public:
  void TearDown() {
    for (auto path : tmp_files_) {
      std::remove(path.c_str());
    }
  }

  void CreateFile(const std::string path, int64_t size) {
    FILE* file = fopen(path.c_str(), "w");
    if (file != nullptr) { tmp_files_.push_back(path); }
    ftruncate(fileno(file), size);
    fclose(file);
  }

 private:
  std::vector<std::string> tmp_files_;
};

#endif
