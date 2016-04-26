cmake_minimum_required(VERSION 2.8)

project(pynumbuf)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${NUMPY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(pynumbuf SHARED numbuf/serialize.cc numbuf/numbuf.cc numbuf/adapters/numpy.cc numbuf/adapters/python.cc)
target_link_libraries(pynumbuf arrow numbuf)
