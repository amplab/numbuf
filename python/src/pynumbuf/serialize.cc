#include "serialize.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API
#include <numpy/arrayobject.h>

#include "adapters/numpy.h"
#include "adapters/python.h"
#include "adapters/numbuf.h"

namespace pynumbuf {

PythonObjectWriter::PythonObjectWriter(arrow::MemoryPool* pool) : pool_(pool) {}

arrow::Status PythonObjectWriter::AssemblePayload(PyObject* value) {
  if(PyArray_Check(value)) {
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
    NumPyToArrow(array, &data_payload_);
    type_ = numbuf::DataType::TENSOR;
    info_ = PyArray_TYPE(array);
  } else if(PyDict_Check(value)) {
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    bool all_children_are_arrays = true;
    while (PyDict_Next(value, &pos, &key, &val)) {
      if(!PyArray_Check(val)) {
        all_children_are_arrays = false;
      }
    }
    if (all_children_are_arrays) {
      NumBufToArrow(value, &data_payload_);
      type_ = numbuf::DataType::NUMBUF;
      info_ = 0;
    } else {
      PyDictToArrow(value, &data_payload_);
      type_ = numbuf::DataType::DICT;
      info_ = 0;
    }
  }
  return arrow::Status::OK();
}

arrow::Status PythonObjectWriter::GetTotalSize(int64_t* size) {
  std::shared_ptr<arrow::RowBatch> header;
  ARROW_RETURN_NOT_OK(numbuf::MakeHeader(type_, 0, 0, &header));
  ARROW_RETURN_NOT_OK(arrow::ipc::GetRowBatchSize(header.get(), size));
  ARROW_RETURN_NOT_OK(arrow::ipc::GetRowBatchSize(data_payload_.get(), &data_size_));
  *size += data_size_;
  return arrow::Status::OK();
}

arrow::Status PythonObjectWriter::Write(arrow::ipc::MemorySource* target, int64_t* metadata_offset) {
  int64_t offset;
  ARROW_RETURN_NOT_OK(arrow::ipc::WriteRowBatch(target, data_payload_.get(), 0, &offset));
  std::shared_ptr<arrow::RowBatch> header;
  ARROW_RETURN_NOT_OK(numbuf::MakeHeader(type_, info_, offset, &header));
  ARROW_RETURN_NOT_OK(arrow::ipc::WriteRowBatch(target, header.get(), data_size_, metadata_offset));
  return arrow::Status::OK();
}

arrow::Status ReadPythonObjectFrom(arrow::ipc::MemorySource* source, int64_t metadata_offset, PyObject** out) {
  std::shared_ptr<arrow::ipc::RowBatchReader> reader;
  ARROW_RETURN_NOT_OK(arrow::ipc::RowBatchReader::Open(source, metadata_offset, &reader));
  auto header_schema = numbuf::MakeHeaderSchema();
  std::shared_ptr<arrow::RowBatch> header;
  reader->GetRowBatch(header_schema, &header);
  int64_t type = dynamic_cast<arrow::Int64Array*>(header->column(0).get())->Value(0);
  int64_t info = dynamic_cast<arrow::Int64Array*>(header->column(1).get())->Value(0);
  int64_t offset = dynamic_cast<arrow::Int64Array*>(header->column(2).get())->Value(0);
  ARROW_RETURN_NOT_OK(arrow::ipc::RowBatchReader::Open(source, offset, &reader));
  if (type == numbuf::DataType::TENSOR) {
    std::shared_ptr<arrow::RowBatch> data;
    auto dtype = numpy_type_to_arrow(info);
    auto schema = numbuf::Tensor::schema(dtype);
    ARROW_RETURN_NOT_OK(reader->GetRowBatch(schema, &data));
    ArrowToNumPy(data, out);
  } else if (type == numbuf::DataType::DICT) {
    std::shared_ptr<arrow::RowBatch> data;
    auto schema = numbuf::Dict::schema();
    ARROW_RETURN_NOT_OK(reader->GetRowBatch(schema, &data));
    ArrowToPyDict(data, out);
  } else if (type == numbuf::DataType::NUMBUF) {
    std::shared_ptr<arrow::RowBatch> data;
    auto schema = numbuf::Numbuf::schema();
    ARROW_RETURN_NOT_OK(reader->GetRowBatch(schema, &data));
    ArrowToNumBuf(data, out);
  }
  return arrow::Status::OK();
}

}
