//
// Created by Eduard Valeyev on 7/10/18.
//

#ifndef TTG_FUTURE_H
#define TTG_FUTURE_H

#include <future>

namespace ttg {

template<typename T>
bool has_value(std::future<T> const& f)
{ return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

template<typename T>
bool has_value(std::shared_future<T> const& f)
{ return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

}

#endif //TTG_FUTURE_H
