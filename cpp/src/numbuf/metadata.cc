#include "metadata.h"
#include "util/utils.h"

namespace numbuf {

std::shared_ptr<arrow::Schema> MakeHeaderSchema() {
  auto type_field = std::make_shared<arrow::Field>("type", std::make_shared<arrow::Int64Type>());
  auto info_field = std::make_shared<arrow::Field>("info", std::make_shared<arrow::Int64Type>());
  auto offset_field = std::make_shared<arrow::Field>("offset", std::make_shared<arrow::Int64Type>());
  return std::shared_ptr<arrow::Schema>(new arrow::Schema({type_field, info_field, offset_field}));
}

arrow::Status MakeHeader(int64_t type, int64_t info, int64_t offset, std::shared_ptr<arrow::RowBatch>* out) {
  auto schema = MakeHeaderSchema();
  auto type_buffer = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  copy_to_buffer({type}, &type_buffer);
  auto type_array = std::make_shared<arrow::Int64Array>(1, type_buffer);
  auto info_buffer = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  copy_to_buffer({info}, &info_buffer);
  auto info_array = std::make_shared<arrow::Int64Array>(1, info_buffer);
  auto offset_buffer = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  copy_to_buffer({offset}, &offset_buffer);
  auto offset_array = std::make_shared<arrow::Int64Array>(1, offset_buffer);
  std::vector<std::shared_ptr<arrow::Array> > fields = {type_array, info_array, offset_array};
  *out = std::make_shared<arrow::RowBatch>(schema, 1, fields);
	return arrow::Status::OK();
}

}
