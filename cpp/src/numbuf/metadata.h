#ifndef NUMBUF_METADATA_H
#define NUMBUF_METADATA_H

#include <arrow/api.h>

namespace numbuf {

struct DataType {
  enum type {
    TENSOR = 0,
    DICT = 1,
    TENSOR_COLLECTION = 2
  };
};

// Return the schema of a header
std::shared_ptr<arrow::Schema> MakeHeaderSchema();

// Construct the header for an object (contains the datatype and the data offset)
arrow::Status MakeHeader(int64_t type, int64_t info, int64_t offset, std::shared_ptr<arrow::RowBatch>* out);

}

#endif
