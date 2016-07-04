// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NUMBUF_UTIL_LOGGING_H
#define NUMBUF_UTIL_LOGGING_H

#include <cstdlib>
#include <iostream>

namespace numbuf {

// Stubbed versions of macros defined in glog/logging.h, intended for
// environments where glog headers aren't available.
//
// Add more as needed.

// Log levels. LOG ignores them, so their values are abitrary.

#define NUMBUF_INFO 0
#define NUMBUF_WARNING 1
#define NUMBUF_ERROR 2
#define NUMBUF_FATAL 3

#define NUMBUF_LOG_INTERNAL(level) numbuf::internal::CerrLog(level)
#define NUMBUF_LOG(level) NUMBUF_LOG_INTERNAL(NUMBUF_##level)

#define NUMBUF_CHECK(condition)                               \
  (condition) ? 0 : ::numbuf::internal::FatalLog(NUMBUF_FATAL) \
                        << __FILE__ << __LINE__ << "Check failed: " #condition " "

#ifdef NDEBUG
#define NUMBUF_DFATAL NUMBUF_WARNING

#else
#define NUMBUF_DFATAL NUMBUF_FATAL

#endif  // NDEBUG

namespace internal {

class NullLog {
 public:
  template <class T>
  NullLog& operator<<(const T& t) {
    return *this;
  }
};

class CerrLog {
 public:
  CerrLog(int severity)  // NOLINT(runtime/explicit)
      : severity_(severity),
        has_logged_(false) {}

  virtual ~CerrLog() {
    if (has_logged_) { std::cerr << std::endl; }
    if (severity_ == NUMBUF_FATAL) { std::exit(1); }
  }

  template <class T>
  CerrLog& operator<<(const T& t) {
    has_logged_ = true;
    std::cerr << t;
    return *this;
  }

 protected:
  const int severity_;
  bool has_logged_;
};

// Clang-tidy isn't smart enough to determine that DCHECK using CerrLog doesn't
// return so we create a new class to give it a hint.
class FatalLog : public CerrLog {
 public:
  FatalLog(int /* severity */)  // NOLINT
      : CerrLog(NUMBUF_FATAL) {}

  [[noreturn]] ~FatalLog() {
    if (has_logged_) { std::cerr << std::endl; }
    std::exit(1);
  }
};

}  // namespace internal

}  // namespace arrow

#endif  // ARROW_UTIL_LOGGING_H
