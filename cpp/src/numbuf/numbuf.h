#ifndef NUMBUF_NUMBUF_H
#define NUMBUF_NUMBUF_H

#include <arrow/api.h>

#include "util/utils.h"
#include "types.h"

namespace numbuf {

class Numbuf {
public:
  Numbuf() : num_bytes_(0), size_(0) {}
  template<class It>
  arrow::Status add_entry(const char* key, size_t key_len, It begin_dims, It end_dims, const arrow::TypePtr& dtype, std::shared_ptr<arrow::Buffer> data) {
    keys_.add_elem(key, key_len);
    dtypes_.push_back(dtype->type);
    dims_.add_elem(begin_dims, end_dims);
    uint8_t* buffer;
    buffer_->Grow(data->size(), &buffer);
    if(offsets_.size() == 0) {
      offsets_.push_back(data->size());
    } else {
      offsets_.push_back(offsets_[offsets_.size()-1] + data->size());
    }
    std::copy(data->data(), data->data() + data->size(), buffer);
    num_bytes_ += data->size();
    size_ += 1;
  }
  static std::shared_ptr<arrow::Schema> schema();
  std::shared_ptr<arrow::RowBatch> content();
private:
  int64_t num_bytes_;
  int64_t size_;
  ArrayBuilder keys_;
  std::vector<int64_t> dtypes_;
  ArrayBuilder dims_;
  std::vector<int32_t> offsets_;
  std::shared_ptr<ElasticBuffer> buffer_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RowBatch> content_;
};

}

#endif