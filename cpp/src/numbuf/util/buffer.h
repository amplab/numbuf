#ifndef NUMBUF_BUFFER_H
#define NUMBUF_BUFFER_H

#include <arrow/util/buffer.h>

namespace numbufold {

static constexpr int64_t MIN_BUFFER_CAPACITY = 1024;

class ElasticBuffer : public arrow::PoolBuffer {
public:
  explicit ElasticBuffer(arrow::MemoryPool* pool = nullptr) : arrow::PoolBuffer(pool) {
    Reserve(MIN_BUFFER_CAPACITY);
  }
  virtual ~ElasticBuffer() {};
  arrow::Status Grow(int64_t amount, uint8_t** data) {
    if (size_ + amount >= capacity_) {
      ARROW_RETURN_NOT_OK(Reserve(3 * capacity_ / 2));
    }
    *data = mutable_data() + size_;
    ARROW_RETURN_NOT_OK(Resize(size_ + amount));
    return arrow::Status::OK();
  }
};

}

#endif
