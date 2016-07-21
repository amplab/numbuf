#include "tensor.h"

namespace numbuf {

Tensor::Tensor(std::initializer_list<int64_t> dims, const arrow::TypePtr& dtype,
    std::shared_ptr<arrow::Buffer> data) : Tensor(dims.begin(), dims.end(), dtype, data) {}

Tensor::Tensor(std::initializer_list<int64_t> dims, const arrow::TypePtr& dtype,
    uint8_t* data, size_t nbytes) : Tensor(dims, dtype, std::make_shared<arrow::Buffer>(data, nbytes)) {}

std::shared_ptr<arrow::Schema> Tensor::schema(const arrow::TypePtr& dtype) {
  auto dtype_field = std::make_shared<arrow::Field>("dtype", INT64_TYPE);
  auto dims_type = std::make_shared<arrow::ListType>(INT64_TYPE);
  auto dims_field = std::make_shared<arrow::Field>("dims", dims_type);
  auto content_type = std::make_shared<arrow::ListType>(dtype);
  auto content_field = std::make_shared<arrow::Field>("content", content_type);
  std::vector<std::shared_ptr<arrow::Field>> fields = {dtype_field, dims_field, content_field};
  return std::make_shared<arrow::Schema>(fields);
}

std::shared_ptr<arrow::RowBatch> Tensor::content() {
  return content_;
}

Tensor Tensor::from_arrow(std::shared_ptr<arrow::RowBatch> batch) {
  int64_t dtype = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0))->Value(0);
  auto dim_content = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(1));
  auto dims = std::dynamic_pointer_cast<arrow::Int64Array>(dim_content->values());
  auto content = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(2));
  auto content_values = std::dynamic_pointer_cast<arrow::PrimitiveArray>(content->values());
  return Tensor(dims->raw_data(), dims->raw_data() + dims->length(), arrow_type(dtype), content_values->data());
}

void Tensor::initialize_content(const arrow::TypePtr& dtype, int64_t size, std::shared_ptr<arrow::Buffer> data) {
  std::shared_ptr<arrow::PoolBuffer> dtype_buffer;
  copy_to_buffer({static_cast<int64_t>(dtype->type)}, &dtype_buffer);
  std::shared_ptr<arrow::PoolBuffer> dim_offsets;
  copy_to_buffer({0, static_cast<int32_t>(dims_.size())}, &dim_offsets);
  std::shared_ptr<arrow::PoolBuffer> dim_buffer;
  copy_to_buffer<int64_t>(dims_, &dim_buffer);
  auto dim_values = std::make_shared<arrow::Int64Array>(static_cast<int32_t>(dims_.size()), dim_buffer);
  auto dims = std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(INT64_TYPE), 1, dim_offsets, dim_values);
  std::shared_ptr<arrow::PoolBuffer> content_offsets;
  copy_to_buffer({0, static_cast<int32_t>(size)}, &content_offsets);
  auto content_values = std::make_shared<arrow::PrimitiveArray>(dtype, size, data);
  auto content = std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(dtype), 1, content_offsets, content_values);
  auto dtype_array = std::make_shared<arrow::Int64Array>(1, dtype_buffer);
  std::vector<std::shared_ptr<arrow::Array>> arrays = {dtype_array, dims, content};
  content_ = std::make_shared<arrow::RowBatch>(schema_, 1, arrays);
}

// TODO(pcm): investigate if there is a performance problem here
const uint8_t* Tensor::data() {
  auto content = std::dynamic_pointer_cast<arrow::ListArray>(content_->column(2));
  auto content_values = std::dynamic_pointer_cast<arrow::PrimitiveArray>(content->values());
  return content_values->data()->data();
}

}
