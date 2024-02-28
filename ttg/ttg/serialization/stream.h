//
// Created by Eduard Valeyev on 5/12/21.
//

#ifndef TTG_SERIALIZATION_STREAM_H
#define TTG_SERIALIZATION_STREAM_H

#include <streambuf>
#include <cstring>

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

  /// streambuf that records vector of address-size pairs
  class iovec_ostreambuf : public std::streambuf {
   public:
    using std::streambuf::streambuf;

    const auto& iovec() const { return iovec_; };

   protected:
    std::streamsize xsputn(const char_type* s, std::streamsize n) override {
      iovec_.emplace_back(s, n);
      return n;
    }

   private:
    std::vector<std::pair<const void*, std::size_t>> iovec_ = {};
  };

  /// streambuf that reads vector of address-size pairs
  class iovec_istreambuf : public std::streambuf {
   public:
    using std::streambuf::streambuf;

    iovec_istreambuf(const std::vector<std::pair<const void*, std::size_t>>& iovec) : iovec_(iovec) {}

   protected:
    std::streamsize xsgetn(char_type* s, std::streamsize max_n) override {
      const auto n = iovec_[current_item_].second;
      if (n > max_n)
        throw std::out_of_range("iovec_istreambuf::xsgetn(dest, max_n): actual size of data exceeds max_n");
      const char* ptr = static_cast<const char*>(iovec_[current_item_].first);
      std::copy(ptr, ptr + n, s);
      return n;
    }

   private:
    std::size_t current_item_ = 0;
    const std::vector<std::pair<const void*, std::size_t>>& iovec_;
  };

  /// streambuf that writes bytes to a buffer in memory
  class byte_ostreambuf : public std::streambuf {
   public:
    using std::streambuf::streambuf;

    byte_ostreambuf(char_type* buffer, std::streamsize buffer_size = std::numeric_limits<std::streamsize>::max()) : buffer_(buffer), cursor_(buffer_), buffer_size_(buffer_size) {}

    // hides basic_streambuf::sputn so can avoid the virtual function dispatch if the compiler is not aggressive enough
    std::streamsize sputn(const char_type* s, std::streamsize n) noexcept {
      return this->xsputn(s, n);
    }

    std::streamsize xsputn(const char_type* s, std::streamsize n) noexcept override final {
      assert((cursor_ - buffer_) + n <= buffer_size_);
      std::memcpy(cursor_, s, n * sizeof(char_type));
      cursor_ += n;
      return n;
    }

   private:
    char_type* buffer_;
    char_type* cursor_;  // current location in buffer_
    std::streamsize buffer_size_;
  };

  /// streambuf that writes bytes to a buffer in memory
  class byte_istreambuf : public std::streambuf {
   public:
    using std::streambuf::streambuf;

    byte_istreambuf(char_type* buffer, std::size_t buffer_size = std::numeric_limits<std::size_t>::max()) : buffer_(buffer), cursor_(buffer_), buffer_size_(buffer_size) {}

    // hides basic_streambuf::sgetn so can avoid the virtual function dispatch if the compiler is not aggressive enough
    std::streamsize sgetn(char_type* s, std::streamsize n) noexcept {
      return this->xsgetn(s, n);
    }

    std::streamsize xsgetn(char_type* s, std::streamsize max_n) noexcept override final {
      const auto n_to_read = std::min(buffer_size_ - (cursor_ - buffer_), max_n);
      std::memcpy(s, cursor_, n_to_read * sizeof(char_type));
      cursor_ += n_to_read;
      return n_to_read;
    }

   private:
    char_type* buffer_;
    char_type* cursor_;  // current location in buffer_
    std::streamsize buffer_size_;
  };

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_STREAM_H
