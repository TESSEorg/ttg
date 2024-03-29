#ifndef TTG_PARSEC_PTR_H
#define TTG_PARSEC_PTR_H

#include <unordered_map>
#include <mutex>

#include "ttg/parsec/ttg_data_copy.h"
#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/task.h"

namespace ttg_parsec {

  // fwd decl
  template<typename T>
  struct ptr;

  namespace detail {
    /* fwd decl */
    template <typename Value>
    inline ttg_data_copy_t *create_new_datacopy(Value &&value);

    struct ptr_impl {
      using copy_type = detail::ttg_data_copy_t;

    private:
      static inline std::unordered_map<ptr_impl*, bool> m_ptr_map;
      static inline std::mutex m_ptr_map_mtx;

      copy_type *m_copy = nullptr;

      void drop_copy() {
        std::cout << "ptr drop_copy " << m_copy << " ref " << m_copy->num_ref() << std::endl;
        if (nullptr != m_copy && 1 == m_copy->drop_ref()) {
          delete m_copy;
        }
        m_copy = nullptr;
      }

      void register_self() {
        /* insert ourselves from the list of ptr */
        std::lock_guard _{m_ptr_map_mtx};
        m_ptr_map.insert(std::pair{this, true});
      }

      void deregister_self() {
        /* remove ourselves from the list of ptr */
        std::lock_guard _{m_ptr_map_mtx};
        if (m_ptr_map.contains(this)) {
          m_ptr_map.erase(this);
        }
      }

    public:
      ptr_impl(copy_type *copy)
      : m_copy(copy)
      {
        register_self();
        m_copy->add_ref();
        std::cout << "ptr copy_obj ref " << m_copy->num_ref() << std::endl;
      }

      copy_type* get_copy() const {
        return m_copy;
      }

      ptr_impl(const ptr_impl& p)
      : m_copy(p.m_copy)
      {
        register_self();
        m_copy->add_ref();
        std::cout << "ptr cpy " << m_copy << " ref " << m_copy->num_ref() << std::endl;
      }

      ptr_impl(ptr_impl&& p)
      : m_copy(p.m_copy)
      {
        register_self();
        p.m_copy = nullptr;
        std::cout << "ptr mov " << m_copy << " ref " << m_copy->num_ref() << std::endl;
      }

      ~ptr_impl() {
        deregister_self();
        drop_copy();
      }

      ptr_impl& operator=(const ptr_impl& p)
      {
        drop_copy();
        m_copy = p.m_copy;
        m_copy->add_ref();
        std::cout << "ptr cpy " << m_copy << " ref " << m_copy->num_ref() << std::endl;
        return *this;
      }

      ptr_impl& operator=(ptr_impl&& p) {
        drop_copy();
        m_copy = p.m_copy;
        p.m_copy = nullptr;
        std::cout << "ptr mov " << m_copy << " ref " << m_copy->num_ref() << std::endl;
        return *this;
      }

      bool is_valid() const {
        return (nullptr != m_copy);
      }

      void reset() {
        drop_copy();
      }

      /* drop all currently registered ptr
       * \note this function is not thread-safe
       *       and should only be called at the
       *       end of the execution, e.g., during finalize.
       */
      static void drop_all_ptr() {
        for(auto it : m_ptr_map) {
          it.first->drop_copy();
        }
      }
    };


    template<typename T>
    ttg_parsec::detail::ttg_data_copy_t* get_copy(ttg_parsec::Ptr<T>& p);
  } // namespace detail

  // fwd decl
  template<typename T, typename... Args>
  Ptr<T> make_ptr(Args&&... args);

  // fwd decl
  template<typename T>
  inline Ptr<std::decay_t<T>> get_ptr(T&& obj);

  template<typename T>
  struct Ptr {

    using value_type = std::decay_t<T>;

  private:
    using copy_type = detail::ttg_data_value_copy_t<value_type>;

    std::unique_ptr<detail::ptr_impl> m_ptr;

    /* only PaRSEC backend functions are allowed to touch our private parts */
    template<typename... Args>
    friend Ptr<T> make_ptr(Args&&... args);
    template<typename S>
    friend Ptr<std::decay_t<S>> get_ptr(S&& obj);
    template<typename S>
    friend detail::ttg_data_copy_t* detail::get_copy(Ptr<S>& p);
    friend ttg::detail::value_copy_handler<ttg::Runtime::PaRSEC>;

    /* only accessible by get_ptr and make_ptr */
    Ptr(detail::ptr_impl::copy_type *copy)
    : m_ptr(new detail::ptr_impl(copy))
    { }

    copy_type* get_copy() const {
      return static_cast<copy_type*>(m_ptr->get_copy());
    }

  public:

    Ptr() = default;

    Ptr(const Ptr& p)
    : Ptr(p.get_copy())
    { }

    Ptr(Ptr&& p) = default;

    ~Ptr() = default;

    Ptr& operator=(const Ptr& p) {
      m_ptr.reset(new detail::ptr_impl(p.get_copy()));
      return *this;
    }

    Ptr& operator=(Ptr&& p) = default;

    value_type& operator*() const {
      return **static_cast<copy_type*>(m_ptr->get_copy());
    }

    value_type& operator->() const {
      return **static_cast<copy_type*>(m_ptr->get_copy());
    }

    bool is_valid() const {
      return m_ptr && m_ptr->is_valid();
    }

    void reset() {
      m_ptr.reset();
    }
  };

#if 0
  namespace detail {
    template<typename Arg>
    inline auto get_ptr(Arg&& obj) {

      for (int i = 0; i < detail::parsec_ttg_caller->data_count; ++i) {
        detail::ttg_data_copy_t *copy = detail::parsec_ttg_caller->copies[i];
        if (nullptr != copy) {
          if (copy->get_ptr() == &obj) {
            bool is_ready = true;
            /* TODO: how can we force-sync host and device? Current data could be on either. */
#if 0
            /* check all tracked device data for validity */
            for (auto it : copy) {
              parsec_data_t *data = *it;
              for (int i = 0; i < parsec_nb_devices; ++i) {
                if (nullptr != data->device_copies[i]) {

                } else {
                  is_ready = false;
                }
              }
            }
#endif // 0
            return std::make_pair(is_ready, std::tuple{ttg_parsec::ptr<std::decay_t<Arg>>(copy)});
          }
        }
      }

      throw std::runtime_error("ttg::get_ptr called on an unknown object!");
    }
  }

  template<typename... Args>
  inline std::pair<bool, std::tuple<ptr<std::decay_t<Args>>...>> get_ptr(Args&&... args) {
    if (nullptr == detail::parsec_ttg_caller) {
      throw std::runtime_error("ttg::get_ptr called outside of a task!");
    }

    bool ready = true;
    auto fn = [&](auto&& arg){
      auto pair = get_ptr(std::forward<decltype(arg)>(arg));
      ready &= pair.first;
      return std::move(pair.second);
    };
    std::tuple<ptr<std::decay_t<Args>>...> tpl = {(fn(std::forward<Args>(args)))...};
    return {ready, std::move(tpl)};
  }
#endif // 0

  template<typename T>
  inline Ptr<std::decay_t<T>> get_ptr(T&& obj) {
    using ptr_type = Ptr<std::decay_t<T>>;
    if (nullptr != detail::parsec_ttg_caller) {
      for (int i = 0; i < detail::parsec_ttg_caller->data_count; ++i) {
        detail::ttg_data_copy_t *copy = detail::parsec_ttg_caller->copies[i];
        if (nullptr != copy) {
          if (copy->get_ptr() == &obj) {
            return ptr_type(copy);
          }
        }
      }
    }
    /* object not tracked, make a new ptr that is now tracked */
    detail::ttg_data_copy_t *copy = detail::create_new_datacopy(obj);
    return ptr_type(copy);
  }

  template<typename T, typename... Args>
  inline Ptr<T> make_ptr(Args&&... args) {
    detail::ttg_data_copy_t *copy = detail::create_new_datacopy(T(std::forward<Args>(args)...));
    return Ptr<T>(copy);
  }

  namespace detail {
    template<typename T>
    inline detail::ttg_data_copy_t* get_copy(ttg_parsec::Ptr<T>& p) {
      return p.get_copy();
    }
  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_PTR_H