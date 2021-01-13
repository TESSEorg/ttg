#ifndef TTG_BASE_KEYMAP_H
#define TTG_BASE_KEYMAP_H

#include <type_traits>
#include "../util/meta.h"
#include "../util/hash.h"

namespace ttg {
  namespace detail {

    /// the default keymap implementation requires ttg::hash{}(key) ... use SFINAE
    /// TODO improve error messaging via more elaborate techniques e.g.
    /// https://gracicot.github.io/tricks/2017/07/01/deleted-function-diagnostic.html
    template <typename keyT, typename Enabler = void>
    struct default_keymap_impl;
    template <typename keyT>
    struct default_keymap_impl<
        keyT, std::enable_if_t<meta::has_ttg_hash_specialization_v<keyT> || meta::is_void_v<keyT>>> {
      default_keymap_impl() = default;
      default_keymap_impl(int world_size) : world_size(world_size) {}

      template <typename Key = keyT>
      std::enable_if_t<!meta::is_void_v<Key>,int>
      operator()(const Key &key) const { return ttg::hash<keyT>{}(key) % world_size; }
      template <typename Key = keyT>
      std::enable_if_t<meta::is_void_v<Key>,int>
      operator()() const { return 0; }

     private:
      int world_size;
    };

  }  // namespace detail

} // namespace ttg

#endif // TTG_BASE_KEYMAP_H
