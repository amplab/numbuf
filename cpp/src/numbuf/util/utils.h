#ifndef ETHER_UTILS_H
#define ETHER_UTILS_H

#include <arrow/api.h>

namespace numbuf {

template <typename It, typename T>
arrow::Status copy_to_buffer(It first, It last, std::shared_ptr<arrow::PoolBuffer>* pool_buffer) {
  arrow::MemoryPool* default_pool = arrow::default_memory_pool();
  auto data = std::make_shared<arrow::PoolBuffer>(default_pool);
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
