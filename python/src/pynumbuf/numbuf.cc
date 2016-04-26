#include <Python.h>
#include <arrow/api.h>
#include <arrow/ipc/memory.h>
#include <arrow/ipc/adapter.h>
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API
#include <numpy/arrayobject.h>

#include <iostream>

#include "numbuf.h"
#include "serialize.h"

extern "C" {

struct shared_data {
  std::shared_ptr<arrow::ipc::MemoryMappedSource> mmap;
  pynumbuf::MemoryMapFixture fixture;
  int64_t metadata_offset;
};

int PyObjectToArrow(PyObject* object, shared_data **result) {
  if (PyCapsule_IsValid(object, "arrow")) {
    *result = static_cast<shared_data*>(PyCapsule_GetPointer(object, "arrow"));
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be a 'call' capsule");
    return 0;
  }
}

PyObject* serialize_object(PyObject* self, PyObject* args) {
  PyObject* value;
  if (!PyArg_ParseTuple(args, "O", &value)) {
    return NULL;
  }
  shared_data* data = new shared_data();

  pynumbuf::PythonObjectWriter writer(arrow::default_memory_pool());
  writer.AssemblePayload(value);
  int64_t size;
  writer.GetTotalSize(&size);

	std::string path("test-serialize");
	data->fixture.CreateFile(path, size);
	arrow::ipc::MemoryMappedSource::Open(path, arrow::ipc::MemorySource::READ_WRITE, &data->mmap);
	writer.Write(data->mmap.get(), &data->metadata_offset);

  return PyCapsule_New(static_cast<void*>(data), "arrow", NULL);
}

PyObject* deserialize_object(PyObject* self, PyObject* args) {
  shared_data *data;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToArrow, &data)) {
    return NULL;
  }

  PyObject* result;
	pynumbuf::ReadPythonObjectFrom(data->mmap.get(), data->metadata_offset, &result);
  return result;
}

static PyMethodDef AetherLibMethods[] = {
 { "serialize_object", serialize_object, METH_VARARGS, "serialize an object to arrow" },
 { "deserialize_object", deserialize_object, METH_VARARGS, "serialize an object to arrow" },
 { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initlibpynumbuf(void) {
  PyObject* m;
  m = Py_InitModule3("libpynumbuf", AetherLibMethods, "Python C Extension for Numbuf");
  import_array();
}

}
