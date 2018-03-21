#ifndef MADTYPES_H_INCL
#define MADTYPES_H_INCL

namespace mra {

    using Level = uint32_t; //< Type of level in adaptive tree (>=0)
    using Translation = uint32_t; //< Type of translation in adaptive tree (>=0)
    using Dimension = uint32_t; //< Number of dimension in function or tensor (>=1)
    using Process = uint32_t; //< MPI process rank (>=0)
    using HashValue = uint32_t; //< Type of hash value used by madness (unsigned)
    
    static constexpr size_t MAX_LEVEL = sizeof(Translation)*8 - 1; //< Maximum level or depth of adaptive refinement
    
    template <typename T, Dimension NDIM> using Coordinate = std::array<T,NDIM>; //< Type of coordinate in NDIM dimensions

    namespace detail {
        // make_array not till c++20 ??
        template <typename T, std::size_t N, typename... Ts>
        static inline auto make_array_crude(const Ts&... t) {
            return std::array<T, N>{{t...}};
        }
        
        // will fail for zero length subtuple
        template <std::size_t Begin, std::size_t End, typename... Ts, std::size_t... I>
        static inline auto subtuple_to_array_of_ptrs_(std::tuple<Ts...>& t, std::index_sequence<I...>) {
            using arrayT = typename std::tuple_element<Begin, std::tuple<Ts*...>>::type;
            return make_array_crude<arrayT, End - Begin>(&std::get<I + Begin>(t)...);
        }

        template <std::size_t Begin, std::size_t End, typename... Ts, std::size_t... I>
        static inline auto subtuple_to_array_of_ptrs_const(const std::tuple<Ts...>& t, std::index_sequence<I...>) {
            using arrayT = typename std::tuple_element<Begin, std::tuple<const Ts*...>>::type;
            return make_array_crude<arrayT, End - Begin>(&std::get<I + Begin>(t)...);
        }
    }

    // BS _const stuff needs cleaning up somehow likely by moving const from outside tuple to inside 

    /// Makes an array of pointers to elements (of same type) in tuple in the open range \c [Begin,End).
    template <std::size_t Begin, std::size_t End, typename... T>
    static inline auto subtuple_to_array_of_ptrs(std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_<Begin, End>(t, std::make_index_sequence<End - Begin>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple in the open range \c [Begin,End).
    template <std::size_t Begin, std::size_t End, typename... T>
    static inline auto subtuple_to_array_of_ptrs_const(const std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_const<Begin, End>(t, std::make_index_sequence<End - Begin>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple
    template <typename... T>
    static inline auto tuple_to_array_of_ptrs(std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_<0, std::tuple_size<std::tuple<T...>>::value>
            (t, std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple
    template <typename... T>
    static inline auto tuple_to_array_of_ptrs_const(const std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_const<0, std::tuple_size<std::tuple<T...>>::value>
            (t, std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>());
    }
}

#endif
