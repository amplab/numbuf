#ifndef NUMBUF_TYPES_H
#define NUMBUF_TYPES_H

#include <arrow/api.h>

#include "util/logging.h"

namespace numbuf {

const auto BOOL_TYPE = std::make_shared<arrow::BooleanType>();

const auto INT8_TYPE = std::make_shared<arrow::Int8Type>();
const auto INT16_TYPE = std::make_shared<arrow::Int16Type>();
const auto INT32_TYPE = std::make_shared<arrow::Int32Type>();
const auto INT64_TYPE = std::make_shared<arrow::Int64Type>();

const auto UINT8_TYPE = std::make_shared<arrow::UInt8Type>();
const auto UINT16_TYPE = std::make_shared<arrow::UInt16Type>();
const auto UINT32_TYPE = std::make_shared<arrow::UInt32Type>();
const auto UINT64_TYPE = std::make_shared<arrow::UInt64Type>();

const auto FLOAT_TYPE = std::make_shared<arrow::FloatType>();
const auto DOUBLE_TYPE = std::make_shared<arrow::DoubleType>();

arrow::TypePtr arrow_type(int64_t dtype);

}

#endif
