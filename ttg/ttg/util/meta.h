#ifndef TTG_UTIL_META_H
#define TTG_UTIL_META_H

#include <functional>
#include <type_traits>

#include "ttg/util/span.h"
#include "ttg/util/typelist.h"
#include "ttg/ptr.h"
#include "ttg/devicescratch.h"

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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // (meta)tuple/typelist/typepack manipulations
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // tuple<Ts...> -> tuple<std::remove_reference_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct nonref_tuple;

    template <typename... Ts>
    struct nonref_tuple<std::tuple<Ts...>> {
      using type = std::tuple<typename std::remove_reference<Ts>::type...>;
    };

    template <typename Tuple>
    using nonref_tuple_t = typename nonref_tuple<Tuple>::type;

    template <typename... TupleTs>
    struct tuple_concat;

    template <typename... Ts>
    struct tuple_concat<std::tuple<Ts...>> {
      using type = std::tuple<Ts...>;
    };

    template <typename... Ts, typename... Us, typename... R>
    struct tuple_concat<std::tuple<Ts...>, std::tuple<Us...>, R...> {
      using type = typename tuple_concat<
          decltype(std::tuple_cat(std::declval<std::tuple<Ts...>>(), std::declval<std::tuple<Us...>>())), R...>::type;
    };

    template <typename... TupleTs>
    using tuple_concat_t = typename tuple_concat<TupleTs...>::type;

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

    // clang-format off
    /// This metafunction applies `Predicate` is true when applied to the last element of `Ts...`, if not empty, else evaluates to `Default`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Default the metafunction result for the case of empty `Ts...`
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_last;

    template <template <typename> typename Predicate, bool Default>
    struct probe_last<Predicate, Default> : public std::bool_constant<Default> {};

    template <template <typename> typename Predicate, bool Default, typename T>
    struct probe_last<Predicate, Default, T> : public std::bool_constant<Predicate<T>::value> {};

    template <template <typename> typename Predicate, bool Default, typename T1, typename... Ts>
    struct probe_last<Predicate, Default, T1, Ts...> : public probe_last<Predicate, Default, Ts...> {};

    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_last<Predicate, Default, std::tuple<Ts...>> : public probe_last<Predicate, Default, Ts...> {};

    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_last<Predicate, Default, ttg::typelist<Ts...>> : public probe_last<Predicate, Default, Ts...> {};

    // clang-format off
    /// This constexpr function applies `Predicate` is true when applied to the last element of `Ts...`, if not empty, else evaluates to `Default`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Default the metafunction result for the case of empty `Ts...`
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, bool Default, typename... Ts>
    constexpr bool probe_last_v = probe_last<Predicate, Default, Ts...>::value;

    // clang-format off
    /// This metafunction checks if `Predicate` is true when applied to the first element of `Ts...`, if not empty, else evaluates to `Default`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Default the metafunction result for the case of empty `Ts...`
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_first;

    template <template <typename> typename Predicate, bool Default>
    struct probe_first<Predicate, Default> : public std::bool_constant<Default> {};

    template <template <typename> typename Predicate, bool Default, typename T1, typename... Ts>
    struct probe_first<Predicate, Default, T1, Ts...> : public std::bool_constant<Predicate<T1>::value> {};

    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_first<Predicate, Default, std::tuple<Ts...>> : public probe_first<Predicate, Default, Ts...> {};

    template <template <typename> typename Predicate, bool Default, typename... Ts>
    struct probe_first<Predicate, Default, ttg::typelist<Ts...>> : public probe_first<Predicate, Default, Ts...> {};

    // clang-format off
    /// This constexpr function checks if `Predicate` is true when applied to the first element of `Ts...`, if not empty, else evaluates to `Default`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Default the metafunction result for the case of empty `Ts...`
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, bool Default, typename... Ts>
    constexpr bool probe_first_v = probe_first<Predicate, Default, Ts...>::value;

    // clang-format off
    /// This metafunction checks if `Predicate` is true for any element of `Ts...`; evaluates to false for the empty `Ts...`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, typename... Ts>
    struct probe_any : std::bool_constant<(Predicate<Ts>::value || ...)> {};

    template <template <typename> typename Predicate, typename... Ts>
    struct probe_any<Predicate, std::tuple<Ts...>> : public probe_any<Predicate, Ts...> {};

    template <template <typename> typename Predicate, typename... Ts>
    struct probe_any<Predicate, ttg::typelist<Ts...>> : public probe_any<Predicate, Ts...> {};

    // clang-format off
    /// This constexpr function checks if `Predicate` is true for any element of `Ts...`; evaluates to false for the empty `Ts...`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, typename... Ts>
    constexpr bool probe_any_v = probe_any<Predicate, Ts...>::value;

    // clang-format off
    /// This metafunction checks if `Predicate` is true for all elements of `Ts...`; evaluates to true for the empty `Ts...`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, typename... Ts>
    struct probe_all : std::bool_constant<(Predicate<Ts>::value && ...)> {};

    template <template <typename> typename Predicate, typename... Ts>
    struct probe_all<Predicate, std::tuple<Ts...>> : public probe_all<Predicate, Ts...> {};

    template <template <typename> typename Predicate, typename... Ts>
    struct probe_all<Predicate, ttg::typelist<Ts...>> : public probe_all<Predicate, Ts...> {};

    // clang-format off
    /// This constexpr function checks if `Predicate` is true for all elements of `Ts...`; evaluates to true for the empty `Ts...`
    /// \tparam Predicate boolean metafunction, i.e. `Predicate<T>::value` is a compile-time boolean
    /// \tparam Ts a type pack
    // clang-format on
    template <template <typename> typename Predicate, typename... Ts>
    constexpr bool probe_all_v = probe_all<Predicate, Ts...>::value;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_Void_v
    // is_void_v = Void or void
    // is_none_void_v
    // is_any_void_v
    // is_last_void_v
    // void_to_Void_t
    // is_any_nonconst_lvalue_reference_v
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    constexpr bool is_Void_v = std::is_same_v<std::decay_t<T>, Void>;

    template <typename T>
    constexpr bool is_void_v = is_Void_v<T> || std::is_void_v<T>;

    template <typename T>
    struct is_void : std::bool_constant<is_void_v<T>> {};

    template <typename T>
    constexpr bool is_nonvoid_v = !is_void_v<T>;

    template <typename T>
    struct is_nonvoid : std::bool_constant<is_nonvoid_v<T>> {};

    template <typename... Ts>
    constexpr bool is_all_void_v = (is_void_v<Ts> && ...);

    template <typename... Ts>
    constexpr bool is_all_void_v<ttg::typelist<Ts...>> = is_all_void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_all_Void_v = (is_Void_v<Ts> && ...);

    template <typename... Ts>
    constexpr bool is_all_Void_v<ttg::typelist<Ts...>> = is_all_Void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_any_void_v = (is_void_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_any_void_v<ttg::typelist<Ts...>> = is_all_void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_any_Void_v = (is_Void_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_any_Void_v<ttg::typelist<Ts...>> = is_any_Void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_none_void_v = !is_any_void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_none_void_v<ttg::typelist<Ts...>> = is_none_void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_none_Void_v = !is_any_Void_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_none_Void_v<ttg::typelist<Ts...>> = is_none_Void_v<Ts...>;

    template <typename... Ts>
    struct is_last_void : probe_last<is_void, false, Ts...> {};

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

    template <typename T>
    constexpr bool is_const_lvalue_reference_v =
        std::is_lvalue_reference_v<T> &&std::is_const_v<std::remove_reference_t<T>>;

    template <typename T>
    struct is_const_lvalue_reference : std::bool_constant<is_const_lvalue_reference_v<T>> {};

    template <typename T>
    constexpr bool is_nonconst_lvalue_reference_v =
        std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>;

    template <typename T>
    struct is_nonconst_lvalue_reference : std::bool_constant<is_nonconst_lvalue_reference_v<T>> {};

    template <typename... Ts>
    constexpr bool is_any_nonconst_lvalue_reference_v = (is_nonconst_lvalue_reference_v<Ts> || ...);

    template <typename... Ts>
    constexpr bool is_any_nonconst_lvalue_reference_v<ttg::typelist<Ts...>> = is_any_nonconst_lvalue_reference_v<Ts...>;

    template <typename... Ts>
    constexpr bool is_any_nonconst_lvalue_reference_v<std::tuple<Ts...>> = is_any_nonconst_lvalue_reference_v<Ts...>;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // device type traits
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    struct is_ptr : std::false_type
    { };

    template<typename T>
    struct is_ptr<ttg::Ptr<T>> : std::true_type
    { };

    template<typename T>
    constexpr bool is_ptr_v = is_ptr<T>::value;

    template<typename T>
    struct is_buffer : std::false_type
    { };

    template<typename T>
    constexpr bool is_buffer_v = is_buffer<T>::value;

    template<typename T>
    struct is_devicescratch : std::false_type
    { };

    template<typename T>
    struct is_devicescratch<ttg::devicescratch<T>> : std::true_type
    { };

    template<typename T>
    struct is_devicescratch<const ttg::devicescratch<T>> : std::true_type
    { };

    template<typename T>
    constexpr bool is_devicescratch_v = is_devicescratch<T>::value;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // typelist metafunctions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// drops N elements from the front
    template <typename Typelist, std::size_t N, typename Enabler = void>
    struct drop_first_n;

    template <typename... Ts>
    struct drop_first_n<std::tuple<Ts...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };

    template <typename... Ts>
    struct drop_first_n<typelist<Ts...>, std::size_t(0)> {
      using type = typelist<Ts...>;
    };

    template <typename T, typename... Ts, std::size_t N>
    struct drop_first_n<std::tuple<T, Ts...>, N, std::enable_if_t<N != 0>> {
      using type = typename drop_first_n<std::tuple<Ts...>, N - 1>::type;
    };

    template <typename T, typename... Ts, std::size_t N>
    struct drop_first_n<typelist<T, Ts...>, N, std::enable_if_t<N != 0>> {
      using type = typename drop_first_n<typelist<Ts...>, N - 1>::type;
    };

    /// take first N elements of a type list
    template <typename Typelist, std::size_t N>
    struct take_first_n;

    template <typename ResultTuple, typename InputTuple, std::size_t N, typename Enabler = void>
    struct take_first_n_helper;

    template <typename... Ts, typename... Us>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<Us...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };
    template <typename... Ts, typename... Us>
    struct take_first_n_helper<typelist<Ts...>, typelist<Us...>, std::size_t(0)> {
      using type = typelist<Ts...>;
    };

    template <typename... Ts, typename U, typename... Us, std::size_t N>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<U, Us...>, N, std::enable_if_t<N != 0>> {
      using type = typename take_first_n_helper<std::tuple<Ts..., U>, std::tuple<Us...>, N - 1>::type;
    };
    template <typename... Ts, typename U, typename... Us, std::size_t N>
    struct take_first_n_helper<typelist<Ts...>, typelist<U, Us...>, N, std::enable_if_t<N != 0>> {
      using type = typename take_first_n_helper<typelist<Ts..., U>, typelist<Us...>, N - 1>::type;
    };

    template <typename... Ts, std::size_t N>
    struct take_first_n<std::tuple<Ts...>, N> {
      using type = typename take_first_n_helper<std::tuple<>, std::tuple<Ts...>, N>::type;
    };

    template <typename... Ts, std::size_t N>
    struct take_first_n<typelist<Ts...>, N> {
      using type = typename take_first_n_helper<typelist<>, typelist<Ts...>, N>::type;
    };

    /// drops N trailing elements from a typelist
    template <typename Typelist, std::size_t N, typename Enabler = void>
    struct drop_last_n;

    template <typename... Ts, std::size_t N>
    struct drop_last_n<std::tuple<Ts...>, N, std::enable_if_t<N <= sizeof...(Ts)>> {
      using type = typename take_first_n<std::tuple<Ts...>, (sizeof...(Ts) - N)>::type;
    };
    template <typename... Ts, std::size_t N>
    struct drop_last_n<typelist<Ts...>, N, std::enable_if_t<N <= sizeof...(Ts)>> {
      using type = typename take_first_n<typelist<Ts...>, (sizeof...(Ts) - N)>::type;
    };

    template <typename... Ts, std::size_t N>
    struct drop_last_n<std::tuple<Ts...>, N, std::enable_if_t<!(N <= sizeof...(Ts))>> {
      using type = std::tuple<>;
    };
    template <typename... Ts, std::size_t N>
    struct drop_last_n<typelist<Ts...>, N, std::enable_if_t<!(N <= sizeof...(Ts))>> {
      using type = typelist<>;
    };

    /// converts a type list to a type list of decayed types, e.g. tuple<Ts...> -> tuple<std::decay_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct decayed_typelist;

    template <typename... Ts>
    struct decayed_typelist<std::tuple<Ts...>> {
      using type = std::tuple<std::decay_t<Ts>...>;
    };
    template <typename... Ts>
    struct decayed_typelist<typelist<Ts...>> {
      using type = typelist<std::decay_t<Ts>...>;
    };

    template <typename Tuple>
    using decayed_typelist_t = typename decayed_typelist<Tuple>::type;

    /// filters out elements of a typelist that do not satisfy the predicate
    template <typename T, template <typename...> typename Pred>
    struct filter;

    template <typename FilteredTypelist, template <typename...> typename Pred, typename... ToBeFilteredTs>
    struct filter_impl;

    template <typename... FilteredTs, template <typename...> typename Pred>
    struct filter_impl<typelist<FilteredTs...>, Pred> {
      using type = typelist<FilteredTs...>;
    };
    template <typename... FilteredTs, template <typename...> typename Pred>
    struct filter_impl<std::tuple<FilteredTs...>, Pred> {
      using type = std::tuple<FilteredTs...>;
    };

    template <typename... FilteredTs, template <typename...> typename Pred, typename U, typename... RestOfUs>
    struct filter_impl<typelist<FilteredTs...>, Pred, U, RestOfUs...>
        : std::conditional_t<Pred<U>::value, filter_impl<typelist<FilteredTs..., U>, Pred, RestOfUs...>,
                             filter_impl<typelist<FilteredTs...>, Pred, RestOfUs...>> {};
    template <typename... FilteredTs, template <typename...> typename Pred, typename U, typename... RestOfUs>
    struct filter_impl<std::tuple<FilteredTs...>, Pred, U, RestOfUs...>
        : std::conditional_t<Pred<U>::value, filter_impl<std::tuple<FilteredTs..., U>, Pred, RestOfUs...>,
                             filter_impl<std::tuple<FilteredTs...>, Pred, RestOfUs...>> {};

    template <typename... Ts, template <typename...> typename Pred>
    struct filter<typelist<Ts...>, Pred> : filter_impl<typelist<>, Pred, Ts...> {};
    template <typename... Ts, template <typename...> typename Pred>
    struct filter<std::tuple<Ts...>, Pred> : filter_impl<std::tuple<>, Pred, Ts...> {};

    template <typename T, template <typename...> typename Pred>
    using filter_t = typename filter<T, Pred>::type;

    template <typename T>
    using drop_void = filter<T, is_nonvoid>;

    template <typename T>
    using drop_void_t = typename drop_void<T>::type;

    template <typename T, typename S, typename U>
    struct replace_nonvoid_helper;

    /* non-void S, replace with U */
    template <typename... Ts, typename S, typename... Ss, typename U, typename... Us>
    struct replace_nonvoid_helper<ttg::typelist<Ts...>, ttg::typelist<S, Ss...>, ttg::typelist<U, Us...>> {
      using type =
          typename replace_nonvoid_helper<ttg::typelist<Ts..., U>, ttg::typelist<Ss...>, ttg::typelist<Us...>>::type;
    };

    /* void S, keep */
    template <typename... Ts, typename... Ss, typename U, typename... Us>
    struct replace_nonvoid_helper<ttg::typelist<Ts...>, ttg::typelist<void, Ss...>, ttg::typelist<U, Us...>> {
      using type = typename replace_nonvoid_helper<ttg::typelist<Ts..., void>, ttg::typelist<Ss...>,
                                                   ttg::typelist<U, Us...>>::type;
    };

    /* empty S, done */
    template <typename... Ts, typename... Us>
    struct replace_nonvoid_helper<ttg::typelist<Ts...>, ttg::typelist<>, ttg::typelist<Us...>> {
      using type = ttg::typelist<Ts...>;
    };

    /* empty U, done */
    template <typename... Ts, typename... Ss>
    struct replace_nonvoid_helper<ttg::typelist<Ts...>, ttg::typelist<Ss...>, ttg::typelist<>> {
      using type = ttg::typelist<Ts..., Ss...>;
    };

    /* empty S and U, done */
    template <typename... Ts>
    struct replace_nonvoid_helper<ttg::typelist<Ts...>, ttg::typelist<>, ttg::typelist<>> {
      using type = ttg::typelist<Ts...>;
    };

    /* Replace the first min(sizeof...(T), sizeof...(U)) non-void types in T with types in U; U does not contain void */
    template <typename T, typename U>
    struct replace_nonvoid;

    template <typename... T, typename... U>
    struct replace_nonvoid<ttg::typelist<T...>, ttg::typelist<U...>> {
      using type = typename replace_nonvoid_helper<ttg::typelist<>, ttg::typelist<T...>, ttg::typelist<U...>>::type;
    };

    template <typename... T, typename... U>
    struct replace_nonvoid<std::tuple<T...>, std::tuple<U...>> {
      using type =
          ttg::meta::typelist_to_tuple_t<typename replace_nonvoid<ttg::typelist<T...>, ttg::typelist<U...>>::type>;
    };

    template <typename T, typename U>
    using replace_nonvoid_t = typename replace_nonvoid<T, U>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Tuple-element type conversions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct void_to_Void_tuple;

    template <typename... Ts>
    struct void_to_Void_tuple<std::tuple<Ts...>> {
      using type = std::tuple<void_to_Void_t<Ts>...>;
    };

    template <typename tupleT>
    using void_to_Void_tuple_t = typename void_to_Void_tuple<std::decay_t<tupleT>>::type;

    template <typename T>
    struct add_lvalue_reference_tuple;

    template <typename... Ts>
    struct add_lvalue_reference_tuple<std::tuple<Ts...>> {
      using type = std::tuple<std::add_lvalue_reference_t<Ts>...>;
    };

    template <typename tupleT>
    using add_lvalue_reference_tuple_t = typename add_lvalue_reference_tuple<tupleT>::type;

    template <typename T>
    struct add_glvalue_reference_tuple;

    template <typename... Ts>
    struct add_glvalue_reference_tuple<std::tuple<Ts...>> {
      using type = std::tuple<std::conditional_t<std::is_const_v<Ts>, std::add_lvalue_reference_t<Ts>,
                                                 std::add_rvalue_reference_t<std::remove_const_t<Ts>>>...>;
    };

    template <typename tupleT>
    using add_glvalue_reference_tuple_t = typename add_glvalue_reference_tuple<tupleT>::type;

    template <typename T, typename... Ts>
    struct none_has_reference {
      static constexpr bool value = !std::is_reference_v<T> && none_has_reference<Ts...>::value;
    };

    template <typename T>
    struct none_has_reference<T> {
      static constexpr bool value = !std::is_reference_v<T>;
    };

    template <typename... T>
    struct none_has_reference<ttg::typelist<T...>> : none_has_reference<T...> {};

    template <>
    struct none_has_reference<ttg::typelist<>> : std::true_type {};

    template <typename... T>
    constexpr bool none_has_reference_v = none_has_reference<T...>::value;

    template <typename T>
    struct is_tuple : std::integral_constant<bool, false> {};

    template <typename... Ts>
    struct is_tuple<std::tuple<Ts...>> : std::integral_constant<bool, true> {};

    template <typename T>
    constexpr bool is_tuple_v = is_tuple<T>::value;

    template <typename>
    struct is_span : std::false_type {};

    template <typename T, std::size_t S>
    struct is_span<ttg::span<T, S>> : std::true_type {};

    template <typename T>
    constexpr bool is_span_v = is_span<T>::value;

    template <template <class> class Pred, typename TupleT, std::size_t I, std::size_t... Is>
    struct predicate_index_seq_helper;

    template <template <class> class Pred, typename T, typename... Ts, std::size_t I, std::size_t... Is>
    struct predicate_index_seq_helper<Pred, std::tuple<T, Ts...>, I, Is...> {
      using seq = std::conditional_t<Pred<T>::value,
                                     typename predicate_index_seq_helper<Pred, std::tuple<Ts...>, I + 1, Is..., I>::seq,
                                     typename predicate_index_seq_helper<Pred, std::tuple<Ts...>, I + 1, Is...>::seq>;
    };

    template <template <class> class Pred, std::size_t I, std::size_t... Is>
    struct predicate_index_seq_helper<Pred, std::tuple<>, I, Is...> {
      using seq = std::index_sequence<Is...>;
    };

    template <typename T>
    struct is_none_void_pred : std::integral_constant<bool, is_none_void_v<T>> {};

    /**
     * An index sequence of nonvoid types in the tuple TupleT
     */
    template <typename TupleT>
    using nonvoid_index_seq = typename predicate_index_seq_helper<is_none_void_pred, TupleT, 0>::seq;

    template <typename T>
    struct is_void_pred : std::integral_constant<bool, is_void_v<T>> {};

    /**
     * An index sequence of void types in the tuple TupleT
     */
    template <typename TupleT>
    using void_index_seq = typename predicate_index_seq_helper<is_void_pred, TupleT, 0>::seq;

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
      template <typename Key, typename Return, typename Enabler = void>
      struct keymap;
      template <typename Key, typename Return>
      struct keymap<Key, Return, std::enable_if_t<!is_void_v<Key>>> {
        using type = std::function<Return(const Key &)>;
      };
      template <typename Key, typename Return>
      struct keymap<Key, Return, std::enable_if_t<is_void_v<Key>>> {
        using type = std::function<Return()>;
      };
      template <typename Key, typename Return = int>
      using keymap_t = typename keymap<Key, Return>::type;

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
      struct input_reducers<std::tuple<valueTs...>> {
        using type = std::tuple<typename input_reducer_type<valueTs>::type...>;
      };
      template <typename... valueTs>
      using input_reducers_t = typename input_reducers<valueTs...>::type;

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // prepare_send_callback_t<Key, Value> = std::function<int(const ttg::span<Key> &, const Value &)> protected against void key
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      template <typename Key, typename Value, typename Enabler = void>
      struct prepare_send_callback;
      template <typename Key, typename Value>
      struct prepare_send_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const ttg::span<const Key> &, const Value &)>;
      };
      template <typename Key, typename Value>
      struct prepare_send_callback<Key, Value, std::enable_if_t<!is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void(const ttg::span<const Key> &)>;
      };
      template <typename Key, typename Value>
      struct prepare_send_callback<Key, Value, std::enable_if_t<is_void_v<Key> && !is_void_v<Value>>> {
        using type = std::function<void(const Value &)>;
      };
      template <typename Key, typename Value>
      struct prepare_send_callback<Key, Value, std::enable_if_t<is_void_v<Key> && is_void_v<Value>>> {
        using type = std::function<void()>;
      };
      template <typename Key, typename Value>
      using prepare_send_callback_t = typename prepare_send_callback<Key, Value>::type;

      template<typename Key, typename Enabler = void>
      struct constraint_callback;

      template<typename Key>
      struct constraint_callback<Key, std::enable_if_t<!is_void_v<Key>>> {
        using type = std::function<bool(const Key&)>;
      };

      template<typename Key>
      struct constraint_callback<Key, std::enable_if_t<is_void_v<Key>>> {
        using type = std::function<bool()>;
      };

      template<typename Key>
      using constraint_callback_t = typename constraint_callback<Key>::type;

    }  // namespace detail

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // check whether a type is iterable
    // Taken from https://en.cppreference.com/w/cpp/types/void_t
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename = void>
    struct is_iterable : std::false_type {};

    // this gets used only when we can call std::begin() and std::end() on that type
    template <typename T>
    struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
        : std::true_type {};

    template <typename T>
    constexpr bool is_iterable_v = is_iterable<T>::value;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // check whether a Callable is invocable with the arguments given as a typelist
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename Callable, typename Typelist>
    constexpr bool is_invocable_typelist_v = false;
    template <typename Callable, typename... Args>
    constexpr bool is_invocable_typelist_v<Callable, ttg::typelist<Args...>> = std::is_invocable_v<Callable, Args...>;
    template <typename ReturnType, typename Callable, typename Typelist>
    constexpr bool is_invocable_typelist_r_v = false;
    template <typename ReturnType, typename Callable, typename... Args>
    constexpr bool is_invocable_typelist_r_v<ReturnType, Callable, ttg::typelist<Args...>> =
        std::is_invocable_r_v<ReturnType, Callable, Args...>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // detects the return result of a Callable when invoked with the arguments given as a typelist
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename Callable, typename Typelist>
    struct invoke_result_typelist {};
    template <typename Callable, typename... Args>
    struct invoke_result_typelist<Callable, ttg::typelist<Args...>> : std::invoke_result<Callable, Args...> {};
    template <class F, class... ArgTypes>
    using invoke_result_typelist_t = typename invoke_result_typelist<F, ArgTypes...>::type;

  }  // namespace meta
}  // namespace ttg

#endif  // TTG_UTIL_META_H
