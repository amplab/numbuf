#ifndef ETHER_UTILS_H
#define ETHER_UTILS_H

#include <arrow/api.h>

#include "buffer.h"

namespace numbuf {

class ArrayBuilder {
public:
  ArrayBuilder() : size_(0), offset_size_(0), value_cursor_(0), value_data_(nullptr), offset_data_(nullptr) {}

  arrow::Status push_initial_offset() {
    int32_t zero = 0;
    offset_data_.Append(zero);
    offset_size_ = 1;
    return arrow::Status::OK();
  }

  template<typename T>
  arrow::Status add_elem(const T* values, size_t len) {
    value_data_.Append(values, len);
    size_ += 1;
    offset_size_ += 1;
    value_cursor_ += len;
    offset_data_.Append(value_cursor_);
    return arrow::Status::OK();
  }

  template<typename It>
  arrow::Status add_elem(It begin, It end) {
    for (auto it = begin; it != end; ++it) {
      value_data_.Append(*it);
    }
    size_ += 1;
    offset_size_ += 1;
    value_cursor_ += std::distance(begin, end);
    offset_data_.Append(value_cursor_);
    return arrow::Status::OK();
  }

  template<typename PrimitiveArray>
  std::shared_ptr<PrimitiveArray> values() {
    return std::make_shared<PrimitiveArray>(size_, value_data_.Finish());
  }

  std::shared_ptr<arrow::Int32Array> offsets() {
    return std::make_shared<arrow::Int32Array>(offset_size_, offset_data_.Finish());
  }

private:
  int64_t size_;
  int32_t value_cursor_;
  int64_t offset_size_;
  arrow::BufferBuilder value_data_;
  arrow::BufferBuilder offset_data_;
};

template <typename It, typename T>
arrow::Status copy_to_buffer(It first, It last, std::shared_ptr<arrow::PoolBuffer>* pool_buffer) {
  auto data = std::make_shared<arrow::PoolBuffer>();
  ARROW_RETURN_NOT_OK(data->Resize(std::distance(first, last) * sizeof(T)));
  std::copy(first, last, reinterpret_cast<T*>(data->mutable_data()));
  *pool_buffer = data;
  return arrow::Status::OK();
}

template <typename T>
arrow::Status copy_to_buffer(const std::vector<T>& content, std::shared_ptr<arrow::PoolBuffer>* pool_buffer) {
  return copy_to_buffer<typename std::vector<T>::const_iterator, T>(content.begin(), content.end(), pool_buffer);
}

template <typename T>
arrow::Status copy_to_buffer(std::initializer_list<T> elements, std::shared_ptr<arrow::PoolBuffer>* pool_buffer) {
  return copy_to_buffer<typename std::initializer_list<T>::const_iterator, T>(elements.begin(), elements.end(), pool_buffer);
}

}

#endif
