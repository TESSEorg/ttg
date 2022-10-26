// to be #include'd within runtime::ttg namespace


template <typename keyT,
          typename DevViewFuncT,
          typename DevKernelFuncT,
          typename DevOutFuncT,
          typename... input_edge_valuesT,
          typename... output_edgesT>
auto make_device_tt(DevViewFuncT &&view_func,
                    DevKernelFuncT &&kernel_func,
                    DevOutFuncT &&out_func,
                    ttg::ExecutionSpace space,
                    const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
                    const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
                    const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_edge_valuesT), "input"),
                    const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT), "output")) {

  // TODO:

  return ttg::Void();
}


namespace detail {
  template<typename FuncT, typename keyT, typename... Args, std::size_t... Is>
  void invoke_unpacked_views(FuncT&& func, const keyT& key, std::tuple<Args...>& views, std::index_sequence<Is...>) {
    func(key, std::get<Is>(views)...);
  }

  template<typename HostT, typename... DevTs, std::size_t... Is>
  void allocate_view_on_device(ttg::View<HostT, DevTs...>& view, std::index_sequence<Is...>) {

    /* TODO: allocate memory on device */

    /* TODO: copy data to device */

    allocate_view_on_device(view, std::index_sequence<Is...>{});
  }

  template<typename... ViewTs, std::size_t I, std::size_t... Is>
  void allocate_on_device(std::tuple<ViewTs...>& views, std::index_sequence<I, Is...>) {

    auto& view = std::get<I>(views);
    allocate_view_on_device(view, std::make_index_sequence<view.size()>());

    allocate_on_device<ViewTs..., Is...>();
  }

  template<typename... ViewTs, std::size_t... Is>
  void allocate_on_device(std::tuple<ViewTs...>& views, std::index_sequence<Is...>) {

  }

} // namespace detail


template <typename keyT,
          typename HostFuncT,
          typename DevViewFuncT,
          typename DevKernelFuncT,
          typename DevOutFuncT,
          typename... input_edge_valuesT,
          typename... output_edgesT>
auto make_device_tt(HostFuncT &&host_func,
                    DevViewFuncT &&view_func,
                    DevKernelFuncT &&kernel_func,
                    DevOutFuncT &&out_func,
                    ttg::ExecutionSpace space,
                    const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
                    const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
                    const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_edge_valuesT), "input"),
                    const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT), "output")) {



  auto taskfn = [=](const keyT& key, auto... args){
    auto views = view_func(key, args...);
    /* 1) allocate memory on device */
    auto device_view = views;
    /* 2) move data from views to device */
    /* 3) call kernel function */
    detail::invoke_unpacked_views(key, views, std::index_sequence_for<decltype(views)>());
    /* 4) move data back out into host objects */
    /* 5) call output function */
    out_func(key, args...);
  };

  return make_tt<keyT>(taskfn, inedges, outedges, innames, outnames);
}

