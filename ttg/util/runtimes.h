//
// Created by Eduard Valeyev on 8/28/18.
//

#ifndef TTG_TRAITS_H
#define TTG_TRAITS_H

namespace ttg {

enum class Runtime {
  PaRSEC, MADWorld
};

template <Runtime R>
struct runtime_traits;

template <>
struct runtime_traits<Runtime::PaRSEC> {
  static constexpr const bool supports_streaming_terminal = false;
  static constexpr const bool supports_async_reduction = false;
};

template <>
struct runtime_traits<Runtime::MADWorld> {
  static constexpr const bool supports_streaming_terminal = true;
  static constexpr const bool supports_async_reduction = true;
};

}

#endif //TTG_TRAITS_H
