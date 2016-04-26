#include "dict.h"

#include <cassert>

namespace numbuf {

const int64_t INITIAL_KEY_CAPACITY = 512;

Dict::Dict(int64_t size) : size_(size), key_capacity_(INITIAL_KEY_CAPACITY), key_size_(0), cur_index_(0) {
  key_buffer_ = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  key_buffer_->Resize(key_capacity_ * sizeof(int8_t));
  offset_buffer_ = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  offset_buffer_->Resize(size * sizeof(int32_t));
  value_buffer_ = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  value_buffer_->Resize(size * sizeof(int64_t));
}

// TODO(pcm): Implement more key and value types (at the moment, the keys must
// be strings and the values must be int64)
std::shared_ptr<arrow::Schema> Dict::schema() {
  auto key_field = std::make_shared<arrow::Field>("key", std::make_shared<arrow::Int8Type>());
  auto offset_field = std::make_shared<arrow::Field>("offsets", std::make_shared<arrow::Int32Type>());
  auto val_field = std::make_shared<arrow::Field>("value", std::make_shared<arrow::Int64Type>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({key_field, offset_field, val_field}));
}

arrow::Status Dict::add_entry(const char* begin, size_t key_len, int64_t value) {
  int8_t* key_data = reinterpret_cast<int8_t*>(key_buffer_->mutable_data());
  int32_t* offset_data = reinterpret_cast<int32_t*>(offset_buffer_->mutable_data());
  int64_t* value_data = reinterpret_cast<int64_t*>(value_buffer_->mutable_data());
  assert(cur_index_ < size_);
  if (key_size_ + key_len >= key_capacity_) {
    key_capacity_ *= 3;
    key_capacity_ /= 2;
    key_buffer_->Resize(key_capacity_ * sizeof(int8_t));
    key_data = reinterpret_cast<int8_t*>(key_buffer_->mutable_data());
  }
  memcpy(key_data + key_size_, begin, key_len);
  key_size_ += key_len;
  offset_data[cur_index_] = key_size_;
  value_data[cur_index_] = value;
  cur_index_ += 1;
  return arrow::Status::OK(); // TODO(pcm): do proper error handling here
}

std::shared_ptr<arrow::RowBatch> Dict::content() {
  auto key_array = std::make_shared<arrow::Int8Array>(key_size_, key_buffer_);
  auto offset_array = std::make_shared<arrow::Int32Array>(size_, offset_buffer_);
  auto val_array = std::make_shared<arrow::Int64Array>(size_, value_buffer_);
  auto schema = Dict::schema();
  std::vector<std::shared_ptr<arrow::Array> > arrays = {key_array, offset_array, val_array};
  return std::make_shared<arrow::RowBatch>(schema, size_, arrays);
}

}
