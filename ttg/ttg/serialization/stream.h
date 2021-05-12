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
    size_t size() const { return m_size; }

   protected:
    std::streamsize xsputn(const char_type* __s, std::streamsize __n) override {
      this->m_size += __n;
      return __n;
    }

   private:
    size_t m_size = 0;
  };
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_STREAM_H
