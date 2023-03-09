#ifndef TTG_DEVICE_SEND_H
#define TTG_DEVICE_SEND_H


namespace ttg {

  namespace detail {

    /* structure holding references to the key and value, awaitable */
    template<typename Key, typename Value>
    struct await_send {
        const Key& m_key;
        Value& m_value;

        await_send(const Key& key, Value& val)
        : m_key(key)
        , m_value(val)
        { }
    };

  } // namespace ttg

} // namespace ttg


#endif // TTG_DEVICE_SEND_H