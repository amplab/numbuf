#include <Python.h>
#include <arrow/api.h>
#include "api.h"

extern "C" {

struct shared_data {
  std::shared_ptr<arrow::RowBatch> pointer;
};

int PyObjectToArrow(PyObject* object, std::shared_ptr<arrow::RowBatch> *result) {
  if (PyCapsule_IsValid(object, "arrow")) {
    shared_data* data = static_cast<shared_data*>(PyCapsule_GetPointer(object, "arrow"));
		*result = data->pointer;
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be a 'call' capsule");
    return 0;
  }
}

PyObject* serialize_object(PyObject* self, PyObject* args) {
  PyObject* dict;
  if (!PyArg_ParseTuple(args, "O", &dict)) {
    return NULL;
  }
  shared_data* data = new shared_data();
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  data->pointer = serialize_dict(dict, pool);
  return PyCapsule_New(static_cast<void*>(data), "arrow", NULL);
}

PyObject* deserialize_object(PyObject* self, PyObject* args) {
  std::shared_ptr<arrow::RowBatch> serialized;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToArrow, &serialized)) {
    return NULL;
  }
  return deserialize_dict(serialized);
}

static PyMethodDef AetherLibMethods[] = {
 { "serialize_object", serialize_object, METH_VARARGS, "serialize an object to arrow" },
 { "deserialize_object", deserialize_object, METH_VARARGS, "serialize an object to arrow" },
 { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initlibether(void) {
  PyObject* m;
  m = Py_InitModule3("libether", AetherLibMethods, "Python C Extension for Aether");
}

}
