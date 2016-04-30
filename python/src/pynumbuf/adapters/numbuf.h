#ifndef PYNUMBUF_NUMBUF_H
#define PYNUMBUF_NUMBUF_H

#include <arrow/api.h>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API
#include <numpy/arrayobject.h>

#include <numbuf/numbuf.h>

namespace pynumbuf {

// TODO(pcm): if the data does not live in shared memory, this will not work, as the Tensor object needs to be kept around
arrow::Status ArrowToNumBuf(std::shared_ptr<arrow::RowBatch> batch, PyObject** out);

arrow::Status NumBufToArrow(PyObject* array, std::shared_ptr<arrow::RowBatch> *out);

}

#endif
