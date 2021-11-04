#ifndef TTG_ATTRIBUTES_H
#define TTG_ATTRIBUTES_H

namespace ttg {


    /**
     * A set of attributes supported by TTG.
     */
    enum class Attribute : size_t {
      PRIORITY = 0,
      PROCESS,
      FINAL,
      SOURCE,
      IMMEDIATE
    };


    /**
     * An attribute value that can store either a fixed value or a function
     * to query that value based on a provided key.
     */
    template<Attribute A, typename KeyT, typename ValueT>
    struct AttributeValue {
      using value_type = std::decay_t<ValueT>;
      using key_type   = std::decay_t<KeyT>;
      using function_type = std::function<value_type(const key_type&)>;
      static constexpr const Attribute attribute_id = A;
    private:
      function_type fn;
      value_type val;
    public:

      AttributeValue() : val(value_type{})
      { }

      template<typename Value_>
      AttributeValue(Value_&& value) : val(std::forward<Value_>(value))
      { }

      template<typename Fn, typename = std::enable_if_t<std::is_invocable_r_v<ValueT, Fn, KeyT>>>
      void set(Fn&& fn) {
        this->fn = std::forward<Fn>(fn);
      }

      void set(value_type val) {
        this->val = val;
        this->fn = nullptr;
      }

      value_type get(const KeyT& key) const {
        return fn ? fn(key) : val;
      }

      value_type operator()(const KeyT& key) const {
        return get(key);
      }

    };


    /* Overload for keys of type void */
    template<Attribute A, typename ValueT>
    struct AttributeValue<A, void, ValueT> {
      using value_type = std::decay_t<ValueT>;
      using function_type = std::function<value_type(void)>;
      static constexpr const Attribute attribute_id = A;
    private:
      function_type fn;
      value_type val;
    public:

      AttributeValue() : val(value_type{})
      { }

      template<typename Value_>
      AttributeValue(Value_&& value) : val(std::forward<Value_>(value))
      { }

      template<typename Fn, typename = std::enable_if_t<std::is_invocable_v<Fn>>>
      void set(Fn&& fn) {
        this->fn = std::forward<Fn>(fn);
      }

      void set(value_type val) {
        this->val = val;
        this->fn = nullptr;
      }

      value_type get(void) const {
        return fn ? fn() : val;
      }

      value_type operator()() const {
        return get();
      }

    };

    /* Overload for keys of type ttg::Void */
    template<Attribute A, typename ValueT>
    struct AttributeValue<A, ttg::Void, ValueT> {
      using value_type = std::decay_t<ValueT>;
      using function_type = std::function<value_type(void)>;
      static constexpr const Attribute attribute_id = A;
    private:
      function_type fn;
      value_type val;
    public:

      AttributeValue() : val(value_type{})
      { }

      template<typename Value_>
      AttributeValue(Value_&& value) : val(std::forward<Value_>(value))
      { }

      template<typename Fn, typename = std::enable_if_t<std::is_invocable_v<Fn>>>
      void set(Fn&& fn) {
        this->fn = std::forward<Fn>(fn);
      }

      void set(value_type val) {
        this->val = val;
        this->fn = nullptr;
      }

      value_type get(void) const {
        return fn ? fn() : val;
      }

      value_type operator()() const {
        return get();
      }

    };

    template<typename KeyT>
    using tt_attribute_set_t = std::tuple<AttributeValue<Attribute::PRIORITY,  KeyT, int32_t>,
                                          AttributeValue<Attribute::PROCESS,   KeyT, int32_t>,
                                          AttributeValue<Attribute::FINAL,     KeyT, bool>,
                                          AttributeValue<Attribute::SOURCE,    KeyT, bool>,
                                          AttributeValue<Attribute::IMMEDIATE, KeyT, bool>>;

} // namespace ttg

#endif // TTG_ATTRIBUTES_H
