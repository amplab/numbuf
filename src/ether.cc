#include <Python.h>
#include <arrow/api.h>
#define PY_ARRAY_UNIQUE_SYMBOL ETHER_ARRAY_API
#include "api.h"

extern "C" {

struct shared_data {
  std::shared_ptr<arrow::ipc::MemoryMappedSource> mmap;
  MemoryMapFixture fixture;
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
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  PyObjectWriter writer(value, pool);
  size_t size = writer.assemble_payload_and_return_size();

  std::string path("test-serialize");
  data->fixture.CreateFile(path, size);
  arrow::ipc::MemoryMappedSource::Open(path, arrow::ipc::MemorySource::READ_WRITE, &data->mmap);

  data->metadata_offset = writer.write_object_and_return_metadata_offset(data->mmap.get());

  return PyCapsule_New(static_cast<void*>(data), "arrow", NULL);
}

PyObject* deserialize_object(PyObject* self, PyObject* args) {
  shared_data *data;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToArrow, &data)) {
    return NULL;
  }

  // return deserialize_dict(serialized);
  // return deserialize_csr(serialized);
  return read_arrow_object(data->mmap.get(), data->metadata_offset);
}

static PyMethodDef AetherLibMethods[] = {
 { "serialize_object", serialize_object, METH_VARARGS, "serialize an object to arrow" },
 { "deserialize_object", deserialize_object, METH_VARARGS, "serialize an object to arrow" },
 { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initlibether(void) {
  PyObject* m;
  m = Py_InitModule3("libether", AetherLibMethods, "Python C Extension for Ether");
  import_array();
}

}
