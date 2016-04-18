#include "api.h"

std::shared_ptr<arrow::Schema> dict_schema() {
  auto key_field = std::make_shared<arrow::Field>("key", std::make_shared<arrow::Int8Type>());
  auto offset_field = std::make_shared<arrow::Field>("offsets", std::make_shared<arrow::Int32Type>());
  auto val_field = std::make_shared<arrow::Field>("value", std::make_shared<arrow::Int64Type>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({key_field, offset_field, val_field}));
}

const int64_t INITIAL_KEYS_CAPACITY = 512;

std::shared_ptr<arrow::RowBatch> serialize_dict(PyObject* dict, arrow::MemoryPool* pool) {
  assert(PyDict_Check(dict));
  Py_ssize_t size = PyDict_Size(dict);

  int64_t keys_capacity = INITIAL_KEYS_CAPACITY;
  auto keys = std::make_shared<arrow::PoolBuffer>(pool);
  keys->Resize(keys_capacity * sizeof(int8_t));
  int8_t* keys_buffer = reinterpret_cast<int8_t*>(keys->mutable_data());
  auto offsets = std::make_shared<arrow::PoolBuffer>(pool);
  offsets->Resize(size * sizeof(int32_t));
  int32_t* offset_buffer = reinterpret_cast<int32_t*>(offsets->mutable_data());
  auto values = std::make_shared<arrow::PoolBuffer>(pool);
  values->Resize(size * sizeof(int64_t));
  int64_t* value_buffer = reinterpret_cast<int64_t*>(values->mutable_data());

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  int64_t cur_entry = 0; // index of the dictionary element we are processing
  int32_t cur_cursor = 0; // how many chars we have added to the keys buffer
  while (PyDict_Next(dict, &pos, &key, &value)) {
    assert(cur_entry < size);
    if (PyString_Check(key)) {
      char* buffer;
      Py_ssize_t len;
      PyString_AsStringAndSize(key, &buffer, &len);
      if (cur_cursor + len >= keys_capacity) {
        keys_capacity *= 2;
        keys->Resize(keys_capacity * sizeof(int8_t));
        keys_buffer = reinterpret_cast<int8_t*>(keys->mutable_data());
      }
      memcpy(keys_buffer + cur_cursor, buffer, len);
      cur_cursor += len;
      offset_buffer[cur_entry] = cur_cursor;
    }
    if (PyInt_Check(value)) {
      value_buffer[cur_entry] = PyInt_AsLong(value);
    }
    cur_entry += 1;
  }
  auto key_array = std::make_shared<arrow::Int8Array>(cur_cursor, keys);
  auto offset_array = std::make_shared<arrow::Int32Array>(static_cast<int32_t>(size), offsets);
  auto val_array = std::make_shared<arrow::Int64Array>(static_cast<int32_t>(size), values);

  auto schema = dict_schema();

  return std::shared_ptr<arrow::RowBatch>(new arrow::RowBatch(schema, size, {key_array, offset_array, val_array}));
}

PyObject* deserialize_dict(std::shared_ptr<arrow::RowBatch> rows) {
  PyObject* dict = PyDict_New();
	size_t size = rows->num_rows();
	auto key_array = dynamic_cast<arrow::Int8Array*>(rows->column(0).get());
  auto offset_array = dynamic_cast<arrow::Int32Array*>(rows->column(1).get());
	auto val_array = dynamic_cast<arrow::Int64Array*>(rows->column(2).get());
  int32_t last_offset = 0;
	for (int i = 0; i < size; ++i) {
    int32_t offset = offset_array->Value(i);
		int32_t len = offset - last_offset;
		const char* buffer = (const char*) (key_array->raw_data() + last_offset);
		PyObject* key = PyString_FromStringAndSize(buffer, len);
		PyDict_SetItem(dict, key, PyInt_FromLong(val_array->Value(i)));
    last_offset = offset;
	}
	return dict;
}
