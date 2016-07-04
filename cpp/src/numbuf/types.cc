#include "types.h"

namespace numbuf {

arrow::TypePtr arrow_type(int64_t dtype) {
  switch(dtype) {
    case arrow::Type::BOOL:
      return BOOL_TYPE;

    case arrow::Type::INT8:
      return INT8_TYPE;
    case arrow::Type::INT16:
      return INT16_TYPE;
    case arrow::Type::INT32:
      return INT32_TYPE;
    case arrow::Type::INT64:
      return INT64_TYPE;

    case arrow::Type::UINT8:
      return UINT8_TYPE;
    case arrow::Type::UINT16:
      return UINT16_TYPE;
    case arrow::Type::UINT32:
      return UINT32_TYPE;
    case arrow::Type::UINT64:
      return UINT64_TYPE;

    case arrow::Type::FLOAT:
      return FLOAT_TYPE;
    case arrow::Type::DOUBLE:
      return DOUBLE_TYPE;
    default:
      NUMBUF_LOG(FATAL) << "unknown type";
      return BOOL_TYPE;
  }
  NUMBUF_LOG(FATAL) << "this cannot happen";
  return BOOL_TYPE;
}

}
