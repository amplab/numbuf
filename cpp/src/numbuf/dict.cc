#include "dict.h"

#include <cassert>

namespace numbufold {

// TODO(pcm): Implement more key and value types (at the moment, the keys must
// be strings and the values must be int64)
std::shared_ptr<arrow::Schema> Dict::schema() {
  auto key_field = std::make_shared<arrow::Field>("key", std::make_shared<arrow::Int8Type>());
  auto offset_field = std::make_shared<arrow::Field>("offsets", std::make_shared<arrow::Int32Type>());
  auto val_field = std::make_shared<arrow::Field>("value", std::make_shared<arrow::Int64Type>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({key_field, offset_field, val_field}));
}

arrow::Status Dict::add_entry(const char* key, size_t key_len, int64_t value) {
  keys_.add_elem(key, key_len);
  int64_t* next_val;
  ARROW_RETURN_NOT_OK(val_buffer_->Grow(sizeof(int64_t), reinterpret_cast<uint8_t**>(&next_val)));
  *next_val = value;
  size_ += 1;
  return arrow::Status::OK();
}

std::shared_ptr<arrow::RowBatch> Dict::content() {
  auto key_array = keys_.values<arrow::Int8Array>();
  auto offset_array = keys_.offsets();
  auto val_array = std::make_shared<arrow::Int64Array>(static_cast<int32_t>(size_), val_buffer_);
  auto schema = Dict::schema();
  std::vector<std::shared_ptr<arrow::Array> > arrays = {key_array, offset_array, val_array};
  return std::make_shared<arrow::RowBatch>(schema, static_cast<int>(size_), arrays);
}

}
