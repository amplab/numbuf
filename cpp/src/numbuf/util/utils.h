#ifndef ETHER_UTILS_H
#define ETHER_UTILS_H

#include <arrow/api.h>

#include "buffer.h"

namespace numbufold {

class ArrayBuilder {
public:
  ArrayBuilder() : size_(0), value_buffer_(new ElasticBuffer()), offset_buffer_(new ElasticBuffer()) {}

  template<typename T>
  arrow::Status add_elem(const T* values, size_t key_len) {
    uint8_t* data;
    ARROW_RETURN_NOT_OK(value_buffer_->Grow(key_len * sizeof(T), &data));
    memcpy(data, values, key_len * sizeof(T));
    int32_t* offset;
    ARROW_RETURN_NOT_OK(offset_buffer_->Grow(sizeof(int32_t), reinterpret_cast<uint8_t**>(&offset)));
    *offset = value_buffer_->size();
    size_ += 1;
    return arrow::Status::OK();
  }

  template<typename It>
  arrow::Status add_elem(It begin, It end) {
    uint8_t* data;
    ARROW_RETURN_NOT_OK(value_buffer_->Grow(std::distance(begin, end) * sizeof(std::iterator_traits<It>::value_type), &data));
    std::copy(begin, end, reinterpret_cast<typename std::iterator_traits<It>::pointer>(data));
    int32_t* offset;
    ARROW_RETURN_NOT_OK(offset_buffer_->Grow(sizeof(int32_t), reinterpret_cast<uint8_t**>(&offset)));
    *offset = value_buffer_->size();
    size_ += 1;
    return arrow::Status::OK();
  }

  template<typename PrimitiveArray>
  std::shared_ptr<PrimitiveArray> values() {
    return std::make_shared<PrimitiveArray>(static_cast<int32_t>(size_), value_buffer_);
  }

  std::shared_ptr<arrow::Int32Array> offsets() {
    return std::make_shared<arrow::Int32Array>(static_cast<int32_t>(size_), offset_buffer_);
  }

private:
  int64_t size_;
  std::shared_ptr<ElasticBuffer> value_buffer_;
  std::shared_ptr<ElasticBuffer> offset_buffer_;
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
