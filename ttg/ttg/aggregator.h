#ifndef TTG_AGGREGATOR_H
#define TTG_AGGREGATOR_H

#include <vector>
#include "ttg/fwd.h"
#include "ttg/edge.h"
#include "ttg/util/meta.h"
#include "ttg/terminal.h"

namespace ttg {

  template<typename ValueT>
  struct Aggregator
  {
    template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs>
    friend class TTG_IMPL_NS::TT;
    using decay_value_type = std::decay_t<ValueT>;
    static constexpr bool value_is_const = std::is_const_v<ValueT>;

    static constexpr size_t short_vector_size = 6; // try to fit the Aggregator into 2 cache lines

  public:
    using value_type = std::conditional_t<value_is_const, std::add_const_t<decay_value_type>, decay_value_type>;

  private:
    struct vector_element_t{
      value_type* value;  // pointer to the value
      void *ptr;          // pointer to implementation-specific data
      vector_element_t() = default;
      vector_element_t(value_type *value, void *ptr)
      : value(value), ptr(ptr)
      { }
    };
    using vector_t = typename std::vector<vector_element_t>;

    template<typename VectorElementT>
    struct Iterator {
    private:

      static constexpr bool iterator_value_is_const = std::is_const_v<VectorElementT>;

      VectorElementT* m_ptr = nullptr;
      using value_t         = std::conditional_t<iterator_value_is_const, std::add_const_t<decay_value_type>, decay_value_type>;
      using reference_t     = std::add_lvalue_reference_t<value_t>;
      using pointer_t       = std::add_pointer_t<value_t>;

    public:

      using value_type        = value_t;
      using reference         = reference_t;
      using pointer           = pointer_t;
      using difference_type   = std::ptrdiff_t;


      template<typename Ptr>
      Iterator(Ptr ptr) : m_ptr(&(*ptr))
      { }

      reference operator*() const { return *m_ptr->value; }

      pointer operator->() { return m_ptr->value; }

      // Prefix increment
      Iterator& operator++() { m_ptr++; return *this; }

      // Postfix increment
      Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

      friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
      friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

    };

  public:

    /* types like std::vector */
    using iterator = std::conditional_t<value_is_const, Iterator<std::add_const_t<vector_element_t>>, Iterator<vector_element_t>>;
    using const_iterator = Iterator<std::add_const_t<vector_element_t>>;
    using size_type = typename vector_t::size_type;
    using pointer = value_type*;
    using reference = std::add_lvalue_reference_t<value_type>;
    using const_reference = std::add_const_t<reference>;
    static constexpr const size_type undef_target = std::numeric_limits<size_type>::max();

    Aggregator() : m_vec()
    { }

    Aggregator(size_type target)
    : m_target(target)
    {
      if (target > short_vector_size) {
        m_vec.reserve(target);
        m_is_dynamic = true;
      } else {
        m_is_dynamic = false;
      }
    }

    Aggregator(const Aggregator&) = default;
    Aggregator(Aggregator&&) = default;

    ~Aggregator() = default;

  private:
    /* Add an element to the aggregator */
    void add_value(value_type& value, void *ptr = nullptr) {
      if (m_is_dynamic) {
        m_vec.emplace_back(&value, ptr);
      } else {
        if (m_size < short_vector_size) {
          m_arr[m_size] = vector_element_t(&value, ptr);
        } else {
          move_to_dynamic();
          m_vec.emplace_back(&value, ptr);
        }
      }
      ++m_size;
    }

    bool has_target() {
      return (m_target != undef_target);
    }

    size_type target() const {
      if (m_target == undef_target) {
        throw std::logic_error("Aggregator has no target defined!");
      }
      return m_target;
    }

    auto data() {
      if (m_is_dynamic) {
        return ttg::span(m_vec.data(), m_size);
      } else {
        return ttg::span(static_cast<vector_element_t*>(&m_arr[0]), m_size);
      }
    }

    void move_to_dynamic() {
      assert(!m_is_dynamic);
      vector_t vec;
      if (has_target()) {
        vec.reserve(m_target);
      } else {
        vec.reserve(m_size);
      }
      /* copies elements into dynamic storage */
      vec.insert(vec.begin(), &m_arr[0], &m_arr[m_size]);
      /* move data into member vector */
      m_vec = std::move(vec);
      m_is_dynamic = true;
    }

    vector_element_t* get_ptr() {
      return (m_is_dynamic) ? m_vec.data() : static_cast<vector_element_t*>(&m_arr[0]);
    }

    const vector_element_t* get_ptr() const {
      return (m_is_dynamic) ? m_vec.data() : static_cast<const vector_element_t*>(&m_arr[0]);
    }

  public:
    reference operator[](size_type i) {
      return (m_is_dynamic) ? m_vec[i] : m_arr[i];
    }

    const_reference operator[](size_type i) const {
      return (m_is_dynamic) ? m_vec[i] : m_arr[i];
    }

    reference at(size_type i) {
      return (m_is_dynamic) ? m_vec.at(i) : m_arr[i];
    }

    const_reference at(size_type i) const {
      return (m_is_dynamic) ? m_vec.at(i) : m_arr[i];
    }

    size_type size() const {
      return m_size;
    }

    iterator begin() {
      return iterator(get_ptr());
    }

    const_iterator begin() const {
      return const_iterator(get_ptr());
    }


    const_iterator cbegin() const {
      return const_iterator(get_ptr());
    }

    iterator end() {
      return iterator(get_ptr() + m_size);
    }

    const_iterator end() const {
      return const_iterator(get_ptr() + m_size);
    }

    const_iterator cend() const {
      return const_iterator(get_ptr() + m_size);
    }
  private:
    std::vector<vector_element_t> m_vec;
    vector_element_t m_arr[short_vector_size];
    size_type m_size = 0;
    size_type m_target = undef_target;
    bool m_is_dynamic = true;
  };

  namespace detail {

    /* Trait to determine if a given type is an aggregator */
    template<typename T>
    struct is_aggregator : std::false_type
    { };

    template<typename ValueT>
    struct is_aggregator<Aggregator<ValueT>> : std::true_type
    { };

    /* Trait to determine if a given type is an aggregator */
    template<typename T>
    constexpr bool is_aggregator_v = is_aggregator<T>::value;

    template<typename KeyT, typename AggregatorT, typename TargetFn>
    struct AggregatorFactory {
      using aggregator_type = AggregatorT;
      using key_type = KeyT;

      AggregatorFactory() : m_targetfn([](){ return aggregator_type::undef_target; })
      { }

      AggregatorFactory(TargetFn fn) : m_targetfn(std::forward<TargetFn>(fn))
      { }

      auto operator()(const key_type& key) const {
        return aggregator_type(m_targetfn(key));
      }

    private:
      TargetFn m_targetfn;
    };


    struct AggregatorTargetProvider {

      AggregatorTargetProvider(std::size_t target = Aggregator<int>::undef_target)
      : m_target(target)
      { }

      template<typename T>
      auto operator()(const T&) const {
        return m_target;
      }
    private:
      std::size_t m_target;
    };

  } // namespace detail

  /* Overload of ttg::Edge with AggregatorFactory value type */
  template<typename KeyT, typename ValueT>
  class Edge<KeyT, Aggregator<ValueT>>
  {

  public:
    /* the underlying edge type */
    using edge_type = ttg::Edge<KeyT, ValueT>;
    using aggregator_type = Aggregator<ValueT>;
    using aggregator_factory_type = std::function<aggregator_type(const KeyT&)>;

    using output_terminal_type = ttg::Out<KeyT, ValueT>;
    using key_type = KeyT;
    using value_type = aggregator_type;

    Edge(edge_type& edge)
    : m_edge(edge)
    , m_aggregator_factory([](const KeyT&){ return aggregator_type(); })
    { }

    template<typename AggregatorFactory>
    Edge(edge_type& edge, AggregatorFactory&& aggregator_factory)
    : m_edge(edge), m_aggregator_factory([=](const KeyT& key){ return aggregator_factory(key); })
    { }

    /* Return reference to the underlying edge */
    edge_type& edge() const {
      return m_edge;
    }

    auto aggregator_factory() const {
      return m_aggregator_factory;
    }

    /// probes if this is already has at least one input
    /// calls the underlying edge.live()
    bool live() const {
      return m_edge.live();
    }

    /// call the underlying edge.set_in()
    void set_in(Out<KeyT, ValueT> *in) const {
      m_edge.set_in(in);
    }

    /// call the underlying edge.set_out()
    void set_out(TerminalBase *out) const {
      m_edge.set_out(out);
    }

    /// call the underlying edge.fire()
    template <typename Key = KeyT, typename Value = ValueT>
    std::enable_if_t<ttg::meta::is_all_void_v<Key, Value>> fire() const {
      m_edge.fire();
    }

  private:
    edge_type& m_edge;
    aggregator_factory_type m_aggregator_factory;
  };

  /// overload for remove_wrapper to expose the underlying value type
  namespace meta {
    template<typename T>
    struct remove_wrapper<Aggregator<T>> {
      using type = T;
    };
  } // namespace meta

  template<typename EdgeT, typename TargetFn, typename = std::enable_if_t<std::is_invocable_v<TargetFn, const typename std::decay_t<EdgeT>::key_type>>>
  auto make_aggregator(EdgeT&& inedge,
                       TargetFn&& targetfn)
  {
    using value_type = typename std::decay_t<EdgeT>::value_type;
    using key_type   = typename std::decay_t<EdgeT>::key_type;
    using fact = typename detail::AggregatorFactory<key_type, Aggregator<value_type>, TargetFn>;
    return Edge<key_type, Aggregator<value_type>>(inedge, fact(std::forward<TargetFn>(targetfn)));
  }

  template<typename EdgeT>
  auto make_aggregator(EdgeT&& inedge,
                       size_t target)
  {
    return make_aggregator(inedge, typename detail::AggregatorTargetProvider(target));
  }

  template<typename EdgeT>
  auto make_aggregator(EdgeT&& inedge)
  {
    using value_type = typename std::decay_t<EdgeT>::value_type;
    using key_type   = typename std::decay_t<EdgeT>::key_type;
    return Edge<key_type, Aggregator<value_type>>(inedge);
  }


} // namespace ttg

#endif // TTG_AGGREGATOR_H
