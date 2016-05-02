cmake_minimum_required(VERSION 2.8)

project(pynumbuf)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${NUMPY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(pynumbuf SHARED src/pynumbuf/serialize.cc src/pynumbuf/numbuf.cc src/pynumbuf/adapters/numpy.cc src/pynumbuf/adapters/python.cc src/pynumbuf/adapters/numbuf.cc)
target_link_libraries(pynumbuf arrow numbuf)

install(TARGETS pynumbuf
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  src/pynumbuf/serialize.h
  src/pynumbuf/numbuf.h
  DESTINATION include/pynumbuf/)

install(TARGETS pynumbuf DESTINATION ${CMAKE_SOURCE_DIR}/pynumbuf/)
