#ifndef PYNUMBUF_SERIALIZE_H
#define PYNUMBUF_SERIALIZE_H

#include <Python.h>
#include <arrow/api.h>
#include <arrow/ipc/memory.h>
#include <arrow/ipc/adapter.h>
#include <numbuf/metadata.h>
#include <numbuf/types.h>
#include <numbuf/tensor.h>

class PythonObjectWriter {
public:
  PythonObjectWriter(arrow::MemoryPool* pool);
  arrow::Status AssemblePayload(PyObject* value);
  arrow::Status GetTotalSize(int64_t* size);
  arrow::Status Write(arrow::ipc::MemorySource* target, int64_t* metadata_offset);
private:
  arrow::MemoryPool* pool_;
  int64_t type_;
  int64_t info_;
  std::shared_ptr<arrow::RowBatch> data_payload_;
  int64_t data_size_;
};

arrow::Status ReadPythonObjectFrom(arrow::ipc::MemoryMappedSource* target, int64_t metadata_offset, PyObject** out);

#endif
