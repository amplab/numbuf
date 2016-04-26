#ifndef PYNUMBUF_PYTHON_H
#define PYNUMBUF_PYTHON_H

#include <Python.h>

#include <arrow/api.h>
#include <numbuf/dict.h>

arrow::Status ArrowToPyDict(std::shared_ptr<arrow::RowBatch> batch, PyObject** out);
arrow::Status PyDictToArrow(PyObject* array, std::shared_ptr<arrow::RowBatch> *out);

#endif
