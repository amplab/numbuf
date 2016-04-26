#ifndef PYNUMBUF_H
#define PYNUMBUF_H

namespace pynumbuf {

class MemoryMapFixture {
 public:
  void TearDown() {
    for (auto path : tmp_files_) {
      std::remove(path.c_str());
    }
  }

  void CreateFile(const std::string path, int64_t size) {
    FILE* file = fopen(path.c_str(), "w");
    if (file != nullptr) { tmp_files_.push_back(path); }
    ftruncate(fileno(file), size);
    fclose(file);
  }

 private:
  std::vector<std::string> tmp_files_;
};

}

#endif
