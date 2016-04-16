class Dict {
  static std::shared_ptr<arrow::Schema> schema();
	std::shared_ptr<arrow::RowBatch> serialize(PyObject* dict);
	PyObject* deserialize(std::shared_ptr<arrow::RowBatch> data);
private:
	arrow::MemoryPool* pool_;
};
