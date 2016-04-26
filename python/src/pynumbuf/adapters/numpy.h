#ifndef PYNUMBUF_NUMPY_H
#define PYNUMBUF_NUMPY_H

#include <arrow/api.h>
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API
#include <numpy/arrayobject.h>

namespace pynumbuf {

// TODO(pcm): if the data does not live in shared memory, this will not work, as the Tensor object needs to be kept around
arrow::Status ArrowToNumPy(std::shared_ptr<arrow::RowBatch> batch, PyObject** out);

arrow::Status NumPyToArrow(PyArrayObject* array, std::shared_ptr<arrow::RowBatch> *out);

arrow::TypePtr numpy_type_to_arrow(int numpy_type);

}

#endif
