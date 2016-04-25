#ifndef NUMBUF_TENSOR_H
#define NUMBUF_TENSOR_H

#include <initializer_list>
#include <arrow/api.h>

#include "util/utils.h"
#include "types.h"

namespace numbuf {

class Tensor {
public:
  Tensor(std::initializer_list<int64_t> dims, const arrow::TypePtr& dtype, std::shared_ptr<arrow::Buffer> data);
  Tensor(std::initializer_list<int64_t> dims, const arrow::TypePtr& dtype, uint8_t* data, size_t nbytes);

  template<class It>
  Tensor(It first, It last, const arrow::TypePtr& dtype, std::shared_ptr<arrow::Buffer> data) {
    dtype_ = dtype;
    int64_t size = 1;
    for (auto it = first; it != last; ++it) {
      dims_.push_back(*it);
      size *= *it;
    }
    schema_ = schema(dtype);
    initialize_content(dtype, size, data);
  }

  static Tensor from_arrow(std::shared_ptr<arrow::RowBatch> batch);
  static std::shared_ptr<arrow::Schema> schema(const arrow::TypePtr& dtype);
  std::shared_ptr<arrow::RowBatch> content();
  arrow::TypePtr dtype() { return dtype_; }
  const uint8_t* data();
  int64_t num_dims() { return dims_.size(); }
  int64_t* dims() { return &dims_[0]; }
private:
  // when this method is invoked, the dims_ vector must already be set
  void initialize_schema(const arrow::TypePtr& dtype);
  // when this method is invoked, the schema must already be initialized
  void initialize_content(const arrow::TypePtr& dtype, int64_t size, std::shared_ptr<arrow::Buffer> data);

  arrow::TypePtr dtype_;
  std::vector<int64_t> dims_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RowBatch> content_;
};

}

#endif
