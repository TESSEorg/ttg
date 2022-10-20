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

  // TODO:
  return ttg::Void();
}

