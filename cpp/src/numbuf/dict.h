#ifndef NUMBUF_DICT_H
#define NUMBUF_DICT_H

#include <initializer_list>
#include <arrow/api.h>

#include "util/utils.h"
#include "types.h"

namespace numbuf {

class Dict {
public:
  Dict(int64_t size);
  arrow::Status add_entry(const char* key, size_t keylen, int64_t val);
  // static std::shared_ptr<arrow::Schema> schema(const arrow::TypePtr& keytype, const arrow::TypePtr& valtype);
  static std::shared_ptr<arrow::Schema> schema();
  std::shared_ptr<arrow::RowBatch> content();
private:
  int64_t size_;
  int64_t key_capacity_;
  int64_t key_size_;
  int64_t cur_index_;
  std::shared_ptr<arrow::PoolBuffer> key_buffer_;
  std::shared_ptr<arrow::PoolBuffer> offset_buffer_;
  std::shared_ptr<arrow::PoolBuffer> value_buffer_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RowBatch> content_;
};

}

#endif
