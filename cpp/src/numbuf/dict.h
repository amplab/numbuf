#ifndef NUMBUF_DICT_H
#define NUMBUF_DICT_H

#include <initializer_list>
#include <arrow/api.h>

#include "util/buffer.h"
#include "util/utils.h"
#include "types.h"

namespace numbuf {

class Dict {
public:
  Dict() : size_(0), val_buffer_(new ElasticBuffer()) {}
  arrow::Status add_entry(const char* key, size_t key_len, int64_t val);
  // static std::shared_ptr<arrow::Schema> schema(const arrow::TypePtr& keytype, const arrow::TypePtr& valtype);
  static std::shared_ptr<arrow::Schema> schema();
  std::shared_ptr<arrow::RowBatch> content();
private:
  int64_t size_;
  ArrayBuilder keys_;
  std::shared_ptr<ElasticBuffer> val_buffer_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RowBatch> content_;
};

}

#endif
