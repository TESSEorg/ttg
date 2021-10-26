#ifndef TTG_UTIL_META_H
#define TTG_UTIL_META_H

#include <functional>
#include <type_traits>

#include "ttg/util/span.h"

namespace ttg {

  class Void;

  namespace meta {

#if __cplusplus >= 201703L
    using std::void_t;
#else
    template <class...>
    using void_t = void;
#endif

    template <typename T>
    using remove_cvr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename Tuple, std::size_t N, typename Enabler = void>
    struct drop_first_n;

    template <typename... Ts>
    struct drop_first_n<std::tuple<Ts...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };

    template <typename T, typename... Ts, std::size_t N>
    struct drop_first_n<std::tuple<T, Ts...>, N, std::enable_if_t<N != 0>> {
      using type = typename drop_first_n<std::tuple<Ts...>, N - 1>::type;
    };

    template <typename ResultTuple, typename InputTuple, std::size_t N, typename Enabler = void>
    struct take_first_n_helper;

    template <typename... Ts, typename... Us>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<Us...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };

    template <typename... Ts, typename U, typename... Us, std::size_t N>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<U, Us...>, N, std::enable_if_t<N != 0>> {
      using type = typename take_first_n_helper<std::tuple<Ts..., U>, std::tuple<Us...>, N - 1>::type;
    };

    template <typename Tuple, std::size_t N>
    struct take_first_n {
      using type = typename take_first_n_helper<std::tuple<>, Tuple, N>::type;
    };

    template <typename Tuple, std::size_t N, typename Enabler = void>
    struct drop_last_n;

    template <typename... Ts, std::size_t N>
    struct drop_last_n<std::tuple<Ts...>, N, std::enable_if_t<N <= sizeof...(Ts)>> {
      using type = typename take_first_n<std::tuple<Ts...>, (sizeof...(Ts) - N)>::type;
    };

    template <typename... Ts, std::size_t N>
    struct drop_last_n<std::tuple<Ts...>, N, std::enable_if_t<!(N <= sizeof...(Ts))>> {
      using type = std::tuple<>;
    };

    // tuple<Ts...> -> tuple<std::remove_reference_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct nonref_tuple;

    template <typename... Ts>
    struct nonref_tuple<std::tuple<Ts...>> {
      using type = std::tuple<typename std::remove_reference<Ts>::type...>;
    };

    template <typename Tuple>
    using nonref_tuple_t = typename nonref_tuple<Tuple>::type;

    // tuple<Ts...> -> tuple<std::decay_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct decayed_tuple;

    template <typename... Ts>
    struct decayed_tuple<std::tuple<Ts...>> {
      using type = std::tuple<typename std::decay<Ts>::type...>;
    };

    template <typename Tuple>
    using decayed_tuple_t = typename decayed_tuple<Tuple>::type;

    template <typename Tuple1, typename Tuple2>
    struct tuple_concat;

    template <typename... Ts, typename... Us>
    struct tuple_concat<std::tuple<Ts...>, std::tuple<Us...>> {
      using type = decltype(std::tuple_cat(std::declval<std::tuple<Ts...>>(), std::declval<std::tuple<Us...>>()));
    };

    template <typename Tuple1, typename Tuple2>
    using tuple_concat_t = typename tuple_concat<Tuple1, Tuple2>::type;

    // filtered_tuple<tuple,p>::type returns tuple with types for which the predicate evaluates to true
    template <typename Tuple, template <typename> typename Predicate>
    struct filtered_tuple;

    namespace detail {
      template <bool>
      struct keep_or_drop {
        template <typename E>
        using type = std::tuple<E>;
      };

      template <>
      struct keep_or_drop<false> {
        template <typename E>
        using type = std::tuple<>;
      };
    }  // namespace detail

    template <template <typename> typename Pred, typename... Es>
    struct filtered_tuple<std::tuple<Es...>, Pred> {
      using type = decltype(std::tuple_cat(
          std::declval<typename detail::keep_or_drop<Pred<Es>::value>::template type<Es>>()...));
    };

    template <typename Tuple, template <typename> typename Pred>
    using filtered_tuple_t = typename filtered_tuple<Tuple, Pred>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_Void_v
    // is_void_v = Void or void
    // is_none_void_v
    // is_any_void_v
    // is_last_void_v
    // void_to_Void_t
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    constexpr bool is_Void_v = std::is_same_v<std::decay_t<T>, Void>;

    template <typename T>
    constexpr bool is_void_v = is_Void_v<T> || std::is_void_v<T>;

    template <typename... Ts>
    constexpr bool is_all_void_v = (is_void_v<Ts> && ...);

    template <typename... Ts>
    constexpr bool is_any_void_v = (is_void_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_any_void_v<std::tuple<Ts...>> = (is_void_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_any_Void_v = (is_Void_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_none_void_v = !is_any_void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_none_Void_v = !is_any_Void_v<Ts...>;

    template <typename... Ts>
    struct is_last_void;

    template <>
    struct is_last_void<> : public std::false_type {};

    template <typename T>
    struct is_last_void<T> : public std::conditional_t<is_void_v<T>, std::true_type, std::false_type> {};

    template <typename T1, typename... Ts>
    struct is_last_void<T1, Ts...> : public is_last_void<Ts...> {};

    template <typename... Ts>
    constexpr bool is_last_void_v = is_last_void<Ts...>::value;

    template <typename T>
    struct void_to_Void {
      using type = T;
    };
    template <>
    struct void_to_Void<void> {
      using type = Void;
    };
    template <typename T>
    using void_to_Void_t = typename void_to_Void<T>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_empty_tuple
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // true if tuple is empty or contains only Void types, e.g. is_empty_tuple<std::tuple<>> or
    // is_empty_tuple<std::tuple<Void>> evaluate to true
    template <typename T, typename Enabler = void>
    struct is_empty_tuple : std::false_type {};

    template <typename... Ts>
    struct is_empty_tuple<std::tuple<Ts...>, std::enable_if_t<(is_Void_v<Ts> && ...)>> : std::true_type {};

    template <typename Tuple>
    inline constexpr bool is_empty_tuple_v = is_empty_tuple<Tuple>::value;

    static_assert(!is_empty_tuple_v<std::tuple<int>>, "ouch");
    static_assert(is_empty_tuple_v<std::tuple<>>, "ouch");
    static_assert(is_empty_tuple_v<std::tuple<Void>>, "ouch");
    static_assert(is_empty_tuple_v<std::tuple<Void, Void, Void>>, "ouch");

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // nonesuch struct from Library Fundamentals V2, source from https://en.cppreference.com/w/cpp/experimental/nonesuch
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct nonesuch {
      ~nonesuch() = delete;
      nonesuch(nonesuch const &) = delete;
      void operator=(nonesuch const &) = delete;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_detected family from Library Fundamentals V2, source from
    // https://en.cppreference.com/w/cpp/experimental/is_detected
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    namespace detail {

      template <class Default, class Enabler, template <class...> class TT, class... Args>
      struct detector {
        using value_t = std::false_type;
        using type = Default;
      };

      template <class Default, template <class...> class TT, class... Args>
      struct detector<Default, void_t<TT<Args...>>, TT, Args...> {
        using value_t = std::true_type;
        using type = TT<Args...>;
      };

    }  // namespace detail

    template <template <class...> class TT, class... Args>
    using is_detected = typename detail::detector<nonesuch, void, TT, Args...>::value_t;

    template <template <class...> class TT, class... Args>
    using detected_t = typename detail::detector<nonesuch, void, TT, Args...>::type;

    template <class Default, template <class...> class TT, class... Args>
    using detected_or = detail::detector<Default, void, TT, Args...>;

    template <template <class...> class TT, class... Args>
    constexpr bool is_detected_v = is_detected<TT, Args...>::value;

    template <class Default, template <class...> class TT, class... Args>
    using detected_or_t = typename detected_or<Default, TT, Args...>::type;

    template <class Expected, template <class...> class TT, class... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<TT, Args...>>;

    template <class Expected, template <class...> class TT, class... Args>
    constexpr bool is_detected_exact_v = is_detected_exact<Expected, TT, Args...>::value;

    template <class To, template <class...> class TT, class... Args>
    using is_detected_convertible = std::is_convertible<detected_t<TT, Args...>, To>;

    template <class To, template <class...> class TT, class... Args>
    constexpr bool is_detected_convertible_v = is_detected_convertible<To, TT, Args...>::value;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // type_printer useful to print types in metaprograms
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct type_printer;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // has_std_hash_specialization_v<T> evaluates to true if std::hash<T> is defined
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enabler = void>
    struct has_std_hash_specialization : std::false_type {};
    template <typename T>
    struct has_std_hash_specialization<
        T, ttg::meta::void_t<decltype(std::declval<std::hash<T>>()(std::declval<const T &>()))>> : std::true_type {};
    template <typename T>
    constexpr bool has_std_hash_specialization_v = has_std_hash_specialization<T>::value;

    namespace detail {

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // send_callback_t<key,value> = std::function<void(const key&, const value&>, protected against void key or value
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Value, typename Enabler = void>
      struct send_callback;
      template <typename Key, typename Value>
      struct send_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const Key &, const Value &)>;
      };
      template <typename Key, typename Value>
      struct send_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void(const Key &)>;
      };
      template <typename Key, typename Value>
      struct send_callback<Key, Value, std::enable_if_t<is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const Value &)>;
      };
      template <typename Key, typename Value>
      struct send_callback<Key, Value, std::enable_if_t<is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void()>;
      };
      template <typename Key, typename Value>
      using send_callback_t = typename send_callback<Key, Value>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // move_callback_t<key,value> = std::function<void(const key&, value&&>, protected against void key or value
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Value, typename Enabler = void>
      struct move_callback;
      template <typename Key, typename Value>
      struct move_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const Key &, Value &&)>;
      };
      template <typename Key, typename Value>
      struct move_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void(const Key &)>;
      };
      template <typename Key, typename Value>
      struct move_callback<Key, Value, std::enable_if_t<is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(Value &&)>;
      };
      template <typename Key, typename Value>
      struct move_callback<Key, Value, std::enable_if_t<is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void()>;
      };
      template <typename Key, typename Value>
      using move_callback_t = typename move_callback<Key, Value>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // broadcast_callback_t<key,value> = std::function<void(const key&, value&&>, protected against void key or value
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Value, typename Enabler = void>
      struct broadcast_callback;
      template <typename Key, typename Value>
      struct broadcast_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const ttg::span<const Key> &, const Value &)>;
      };
      template <typename Key, typename Value>
      struct broadcast_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void(const ttg::span<const Key> &)>;
      };
      template <typename Key, typename Value>
      struct broadcast_callback<Key, Value, std::enable_if_t<is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const Value &)>;
      };
      template <typename Key, typename Value>
      struct broadcast_callback<Key, Value, std::enable_if_t<is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void()>;
      };
      template <typename Key, typename Value>
      using broadcast_callback_t = typename broadcast_callback<Key, Value>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // setsize_callback_t<key> = std::function<void(const keyT &, std::size_t)> protected against void key
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Enabler = void>
      struct setsize_callback;
      template <typename Key>
      struct setsize_callback<Key, std::enable_if_t<!is_void_v<Key>>> {
        using type = std::function<void(const Key &, std::size_t)>;
      };
      template <typename Key>
      struct setsize_callback<Key, std::enable_if_t<is_void_v<Key>>> {
        using type = std::function<void(std::size_t)>;
      };
      template <typename Key>
      using setsize_callback_t = typename setsize_callback<Key>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // finalize_callback_t<key> = std::function<void(const keyT &)> protected against void key
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Enabler = void>
      struct finalize_callback;
      template <typename Key>
      struct finalize_callback<Key, std::enable_if_t<!is_void_v<Key>>> {
        using type = std::function<void(const Key &)>;
      };
      template <typename Key>
      struct finalize_callback<Key, std::enable_if_t<is_void_v<Key>>> {
        using type = std::function<void()>;
      };
      template <typename Key>
      using finalize_callback_t = typename finalize_callback<Key>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // keymap_t<key,value> = std::function<int(const key&>, protected against void key
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Enabler = void>
      struct keymap;
      template <typename Key>
      struct keymap<Key, std::enable_if_t<!is_void_v<Key>>> {
        using type = std::function<int(const Key &)>;
      };
      template <typename Key>
      struct keymap<Key, std::enable_if_t<is_void_v<Key>>> {
        using type = std::function<int()>;
      };
      template <typename Key>
      using keymap_t = typename keymap<Key>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // input_reducers_t<valueTs...> = std::tuple<
      //   std::function<std::decay_t<input_valueTs>(std::decay_t<input_valueTs> &&, std::decay_t<input_valueTs>
      //   &&)>...>
      // protected against void valueTs
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename T, typename Enabler = void>
      struct input_reducer_type;
      template <typename T>
      struct input_reducer_type<T, std::enable_if_t<!is_void_v<T>>> {
        using type = std::function<void(std::decay_t<T> &, const std::decay_t<T> &)>;
      };
      template <typename T>
      struct input_reducer_type<T, std::enable_if_t<is_void_v<T>>> {
        using type = std::function<void()>;
      };
      template <typename... valueTs>
      struct input_reducers {
        using type = std::tuple<typename input_reducer_type<valueTs>::type...>;
      };
      template <typename... valueTs>
      using input_reducers_t = typename input_reducers<valueTs...>::type;

    }  // namespace detail

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // check whether a type is iterable
    // Taken from https://en.cppreference.com/w/cpp/types/void_t
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename = void>
    struct is_iterable : std::false_type {};

    // this gets used only when we can call std::begin() and std::end() on that type
    template <typename T>
    struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                      decltype(std::end(std::declval<T>()))
                                    >
                      > : std::true_type {};

    template <typename T>
    constexpr bool is_iterable_v = is_iterable<T>::value;
  } // namespace meta
}  // namespace ttg

#endif  // TTG_UTIL_META_H
