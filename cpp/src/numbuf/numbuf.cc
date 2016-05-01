#include "numbuf.h"

namespace numbuf {

std::shared_ptr<arrow::Schema> Numbuf::schema() {
  auto key_field = std::make_shared<arrow::Field>("key", std::make_shared<arrow::Int8Type>());
  auto key_offset_field = std::make_shared<arrow::Field>("key_offsets", std::make_shared<arrow::Int32Type>());
  auto dtype_field = std::make_shared<arrow::Field>("dtype", INT64_TYPE);
  auto dims_type = std::make_shared<arrow::ListType>(INT64_TYPE);
  auto dims_field = std::make_shared<arrow::Field>("dims", dims_type);
  auto content_type = std::make_shared<arrow::ListType>(UINT8_TYPE);
  auto content_field = std::make_shared<arrow::Field>("content", content_type);
  std::vector<std::shared_ptr<arrow::Field>> fields = {key_field, key_offset_field, dtype_field, dims_field, content_field};
  return std::make_shared<arrow::Schema>(fields);
}

std::shared_ptr<arrow::RowBatch> Numbuf::content() {
  auto key_array = keys_.values<arrow::Int8Array>();
  auto key_offset_array = keys_.offsets();
  std::shared_ptr<arrow::PoolBuffer> dtype_buffer;
  copy_to_buffer(dtypes_, &dtype_buffer);
  auto dtype_array = std::make_shared<arrow::Int64Array>(size_, dtype_buffer);
  auto dim_offsets = dims_.offsets();
  auto dim_values = dims_.values<arrow::Int64Array>();
  auto dims = std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(INT64_TYPE), size_, dim_offsets->data(), dim_values);

  std::shared_ptr<arrow::PoolBuffer> offsets_buffer;
  copy_to_buffer(offsets_, &offsets_buffer);

  auto content_values = std::make_shared<arrow::PrimitiveArray>(UINT8_TYPE, num_bytes_, data_.Finish());
  auto content = std::make_shared<arrow::ListArray>(std::make_shared<arrow::ListType>(UINT8_TYPE), size_, offsets_buffer, content_values);

  std::vector<std::shared_ptr<arrow::Array>> arrays = {key_array, key_offset_array, dtype_array, dims, content};
  auto schema = Numbuf::schema();
  return std::make_shared<arrow::RowBatch>(schema, size_, arrays);
}

}
