//
// Created by Eduard Valeyev on 5/12/21.
//

#ifndef TTG_SERIALIZATION_STREAM_H
#define TTG_SERIALIZATION_STREAM_H

#include <streambuf>

namespace ttg::detail {

  /// streambuf that counts bytes
  class counting_streambuf : public std::streambuf {
   public:
    using std::streambuf::streambuf;

    /// @return the size of data put into `*this`
    size_t size() const { return size_; }

   protected:
    std::streamsize xsputn(const char_type* s, std::streamsize n) override {
      this->size_ += n;
      return n;
    }

   private:
    size_t size_ = 0;
  };
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_STREAM_H
