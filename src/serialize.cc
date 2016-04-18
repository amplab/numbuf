#include <arrow/type.h>
#include <arrow/builder.h>
#include <arrow/api.h>
#include <arrow/ipc/memory.h>
#include <arrow/ipc/adapter.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL ETHER_ARRAY_API

#include "api.h"

#include <iostream>



/*
StructType array_metadata(PyArrayObject* array, MemoryPool* pool) {
	std::shared_ptr<ArrayBuilder> builder;
	MakeBuilder(pool, TypePtr(new ListType(TypePtr(new Int64Type())))); // assert that everything is ok
	shared_ptr<ListBuilder> builder = std::dynamic_point_cast<ListBuilder>(builder);
	builder->Append(dim)



  auto dtype_field = std::make_shared<Field>("dtype", TypePtr(new Int64Type()));
	ListType dim_type(dtype_field);
  auto dim_field = std::make_shared<Field>("dims", dim_type);
	PyArray_TYPE(array);
	PyArray_DIM(array, i)
  std::vector<std::shared_ptr<Field> > fields = {dtype_field, dim_field};
  return StructType(fields);
}

void serialize_array(PyArrayObject* array, MemorySegmentPool* pool) {
  StructType metadata =
}
*/

#define DEFINE_BUILDER_CONSTRUCTOR(TYPE, CapType)                                            \
  std::shared_ptr<arrow::CapType##Builder> make_##TYPE##_builder(arrow::MemoryPool* pool) {  \
    std::shared_ptr<arrow::ArrayBuilder> result;                                             \
    arrow::TypePtr type = arrow::TypePtr(new arrow::CapType##Type());                        \
    arrow::MakeBuilder(pool, type, &result);                                                 \
    return std::dynamic_pointer_cast<arrow::CapType##Builder>(result);                       \
  }

DEFINE_BUILDER_CONSTRUCTOR(string, String);
DEFINE_BUILDER_CONSTRUCTOR(int64, Int64);

std::shared_ptr<arrow::Schema> make_header_schema() {
  auto header_field = std::make_shared<arrow::Field>("header", std::make_shared<arrow::Int64Type>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({header_field}));
}

// TODO: clean up
std::shared_ptr<arrow::RowBatch> make_header(EtherType type, int64_t metadata_offset, arrow::MemoryPool* pool) {
  auto schema = make_header_schema();
  auto header_data = std::make_shared<arrow::PoolBuffer>(pool);
  header_data->Resize(2 * sizeof(int64_t));
  int64_t* header_buffer = reinterpret_cast<int64_t*>(header_data->mutable_data());
  header_buffer[0] = type;
  header_buffer[1] = metadata_offset;
  auto header_array = std::make_shared<arrow::Int64Array>(2, header_data);
	return std::shared_ptr<arrow::RowBatch>(new arrow::RowBatch(schema, 2, {header_array}));
}

PyObjectWriter::PyObjectWriter(PyObject* python_object, arrow::MemoryPool* pool)
  : python_object_(python_object), pool_(pool) {}

int64_t PyObjectWriter::assemble_payload_and_return_size() {
  int64_t result = 0;
  if (PyDict_Check(python_object_)) {
    auto header = make_header(DICT_TYPE, 0, pool_);
    data_payload_ = serialize_dict(python_object_, pool_);
    arrow::ipc::GetRowBatchSize(data_payload_.get(), &data_payload_size_);
    result += data_payload_size_;
    arrow::ipc::GetRowBatchSize(header.get(), &metadata_size_);
    result += metadata_size_;
  }
  /*
  if (PyArray_Check(python_object_)) {
    auto header = make_header(ARRAY_TYPE, 0, pool_);
    auto array_header = make_array_header((PyArrayObject*)python_object_, 0, pool_);
    data_payload_ = serialize_array((PyArrayObject*) python_object_, pool_);
    arrow::ipc::GetRowBatchSize(data_payload_.get(), &data_payload_size_);
    result += data_payload_size_;
    arrow::ipc::GetRowBatchSize(array_header.get(), &data_header_size_);
    result += data_header_size_;
    arrow::ipc::GetRowBatchSize(header.get(), &metadata_size_);
    result += metadata_size_;
  }
  */
  else if (is_csr_matrix(python_object_)) {
    auto header = make_header(CSR_MATRIX, 0, pool_);
    auto csr_header = make_csr_matrix_header(python_object_, 0, pool_);
    data_payload_ = serialize_csr(python_object_, pool_);
    arrow::ipc::GetRowBatchSize(data_payload_.get(), &data_payload_size_);
    result += data_payload_size_;
    arrow::ipc::GetRowBatchSize(csr_header.get(), &data_header_size_);
    result += data_header_size_;
    arrow::ipc::GetRowBatchSize(header.get(), &metadata_size_);
    result += metadata_size_;
  }
  return result;
}

int64_t PyObjectWriter::write_object_and_return_metadata_offset(arrow::ipc::MemorySource* target) {
  int64_t data_offset;
  arrow::ipc::WriteRowBatch(target, data_payload_.get(), 0, &data_offset);
  int64_t metadata_offset;
  if (PyDict_Check(python_object_)) {
    auto header = make_header(DICT_TYPE, data_offset, pool_);
    arrow::ipc::WriteRowBatch(target, header.get(), data_payload_size_, &metadata_offset);
  } else if (PyArray_Check(python_object_)) {
    int64_t array_header_offset;
    auto array_header = make_array_header((PyArrayObject*) python_object_, data_offset, pool_);
    arrow::ipc::WriteRowBatch(target, array_header.get(), data_payload_size_, &array_header_offset);
    auto header = make_header(ARRAY_TYPE, array_header_offset, pool_);
    arrow::ipc::WriteRowBatch(target, header.get(), data_payload_size_ + data_header_size_, &metadata_offset);
  } else if (is_csr_matrix(python_object_)) {
    int64_t csr_header_offset;
    auto csr_header = make_csr_matrix_header(python_object_, data_offset, pool_);
    arrow::ipc::WriteRowBatch(target, csr_header.get(), data_payload_size_, &csr_header_offset);
    auto header = make_header(CSR_MATRIX, csr_header_offset, pool_);
    arrow::ipc::WriteRowBatch(target, header.get(), data_payload_size_ + data_header_size_, &metadata_offset);
  }
  return metadata_offset;
}

PyObject* read_arrow_object(arrow::ipc::MemorySource* source, int64_t metadata_offset) {
  std::shared_ptr<arrow::ipc::RowBatchReader> reader;
  arrow::Status s = arrow::ipc::RowBatchReader::Open(source, metadata_offset, &reader);
  assert(s.ok());
  auto header_schema = make_header_schema();
  std::shared_ptr<arrow::RowBatch> header;
  s = reader->GetRowBatch(header_schema, &header);
  assert(s.ok());
  const int64_t* header_data = dynamic_cast<arrow::Int64Array*>(header->column(0).get())->raw_data();
  int64_t type = header_data[0];
  std::shared_ptr<arrow::RowBatch> data;
  s = arrow::ipc::RowBatchReader::Open(source, header_data[1], &reader);
  assert(s.ok());
  switch (type) {
    case DICT_TYPE: {
      auto data_header = dict_schema();
      s = reader->GetRowBatch(data_header, &data);
      assert(s.ok());
      return deserialize_dict(data);
    }
    case CSR_MATRIX: {
      auto data_header = csr_sparse_header_schema();
      s = reader->GetRowBatch(data_header, &data);
      assert(s.ok());
      const int64_t* data_header_content = dynamic_cast<arrow::Int64Array*>(data->column(0).get())->raw_data();
      int64_t data_offset = data_header_content[2];
      s = arrow::ipc::RowBatchReader::Open(source, data_offset, &reader);
      assert(s.ok());
      auto content_header = csr_sparse_schema();
      std::shared_ptr<arrow::RowBatch> content; // TODO: the distinction between content and data here is horrible
      s = reader->GetRowBatch(content_header, &content);
      assert(s.ok());
      return deserialize_csr(content, data_header_content[0], data_header_content[1]);
    }
    break;
  }
}

/*
void write_python_object(PyObject* python_object, MemoryPool* pool, arrow::ipc::MemorySource* target, int64_t* metadata_offset) {
  PyObjectWriter writer(python_object, pool);
  int64_t size = writer.assemble_payload_and_return_size();

}
*/
