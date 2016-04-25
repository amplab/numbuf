#include "tensor.h"

#include <iostream>

int main() {
  std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  auto buffer = std::make_shared<arrow::PoolBuffer>(arrow::default_memory_pool());
  numbuf::copy_to_buffer(v, &buffer);
  numbuf::Tensor tensor({2, 2}, numbuf::DOUBLE_TYPE, buffer);

  auto schema = numbuf::Tensor::schema(numbuf::DOUBLE_TYPE);
  auto content = tensor.content();

  numbuf::Tensor result = numbuf::Tensor::from_arrow(content);

  const double* data = reinterpret_cast<const double*>(result.data());

  for (int i = 0; i < 4; ++i) {
    std::cout << "entry " << i << " " << data[i] << std::endl;
  }

  std::cout << "num dims " << result.num_dims() << std::endl;
}
