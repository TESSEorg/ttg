//
// Created by Eduard Valeyev on 10/21/22.
//

#ifndef TTG_UTIL_MULTIINDEX_H
#define TTG_UTIL_MULTIINDEX_H

namespace ttg {

  template <std::size_t Rank, typename Int = int>
  struct MultiIndex {
    static constexpr const std::size_t max_index = 1 << 21;
    static constexpr const std::size_t max_index_square = max_index * max_index;
    MultiIndex() = default;
    template <typename Integer, typename = std::enable_if_t<std::is_integral_v<Int>>>
    MultiIndex(std::initializer_list<Integer> ilist) {
      std::copy(ilist.begin(), ilist.end(), data_.begin());
      assert(valid());
    }
    template <typename... Ints, typename = std::enable_if_t<(std::is_integral_v<Ints> && ...)>>
    MultiIndex(Ints... ilist) : data_{{static_cast<Int>(ilist)...}} {
      assert(valid());
    }
    explicit MultiIndex(std::size_t hash) {
      static_assert(Rank == 1 || Rank == 2 || Rank == 3,
                    "MultiIndex<Rank>::MultiIndex(hash) only implemented for Rank={1,2,3}");
      if (Rank == 1) {
        assert(hash < max_index);
        (*this)[0] = hash;
      }
      if (Rank == 2) {
        (*this)[0] = hash / max_index;
        (*this)[1] = hash % max_index;
      } else if (Rank == 3) {
        (*this)[0] = hash / max_index_square;
        (*this)[1] = (hash % max_index_square) / max_index;
        (*this)[2] = hash % max_index;
      }
    }
    std::size_t hash() const {
      static_assert(Rank == 1 || Rank == 2 || Rank == 3, "MultiIndex<Rank>::hash only implemented for Rank={1,2,3}");
      if constexpr (Rank == 1)
        return (*this)[0];
      else if constexpr (Rank == 2) {
        return (*this)[0] * max_index + (*this)[1];
      } else if constexpr (Rank == 3) {
        return ((*this)[0] * max_index + (*this)[1]) * max_index + (*this)[2];
      }
    }

    const auto &operator[](std::size_t idx) const {
      if (idx >= Rank) assert(idx < Rank);
      return data_[idx];
    }

   private:
    bool valid() {
      bool result = true;
      for (const auto &idx : data_) {
        result = result && (idx < max_index);
      }
      return result;
    }

    std::array<Int, Rank> data_;
  };

  template <std::size_t Rank>
  std::ostream &operator<<(std::ostream &os, const MultiIndex<Rank> &key) {
    os << "{";
    for (size_t i = 0; i != Rank; ++i) os << key[i] << (i + 1 != Rank ? "," : "");
    os << "}";
    return os;
  }

}  // namespace ttg

#endif  // TTG_UTIL_MULTIINDEX_H
