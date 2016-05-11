cmake_minimum_required(VERSION 2.8)

project(pynumbuf)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${NUMPY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(ARROW_INCLUDE_DIR "" CACHE FILEPATH "Path to arrow includes")
include_directories(SYSTEM ${ARROW_INCLUDE_DIR})
set(ARROW_LIBRARY_DIR "" CACHE FILEPATH "Path to arrow library")
add_library(arrow SHARED IMPORTED)
set_target_properties(arrow PROPERTIES IMPORTED_LOCATION "${ARROW_LIBRARY_DIR}")

add_library(pynumbuf SHARED src/pynumbuf/serialize.cc src/pynumbuf/numbuf.cc src/pynumbuf/adapters/numpy.cc src/pynumbuf/adapters/python.cc)
target_link_libraries(pynumbuf arrow numbuf)

install(TARGETS pynumbuf
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  src/pynumbuf/serialize.h
  src/pynumbuf/numbuf.h
  DESTINATION include/pynumbuf/)
