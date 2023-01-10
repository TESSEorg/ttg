// clang-format off
#ifndef TTG_MAKE_DEVICE_TT_H
#define TTG_MAKE_DEVICE_TT_H

// to be #include'd within runtime::ttg namespace


namespace detail {
#ifdef TTG_HAVE_CUDA
  inline thread_local cudaStream_t* ts_stream = nullptr;
#endif // TTG_HAVE_CUDA

  template<typename FuncT, typename keyT, typename... Args, std::size_t... Is>
  inline void invoke_with_unpacked_views(FuncT&& func, const keyT& key, std::tuple<Args...>& views, std::index_sequence<Is...>) {
#ifdef TTG_HAVE_CUDA
    func(key, std::get<Is>(views)..., ts_stream);
#else // TTG_HAVE_CUDA
    func(key, std::get<Is>(views)...);
#endif // TTG_HAVE_CUDA
  }

  /* TODO: extract host objects from views */
  template<typename ViewTupleT>
  struct host_obj_type;

  template<typename... ViewTs>
  struct host_obj_type<std::tuple<ViewTs...>> {
    using type = std::tuple<typename ViewTs::host_type...>;
  };

  template<typename ViewTupleT>
  using host_obj_type_t = typename host_obj_type<ViewTupleT>::type;

  template<typename FuncT, typename keyT, typename... ViewTs, typename  std::size_t... Is>
  inline void invoke_out_with_unpacked_views(FuncT&& func, const keyT& key, std::tuple<ViewTs...> views, std::index_sequence<Is...>) {
    func(key, std::get<Is>(views).get_host_object()...);
  }

  template<typename HostT, typename... DevTs, std::size_t I, std::size_t... Is>
  inline void create_view_on_device(const ttg::View<HostT, DevTs...>& view,
                                    std::tuple<ttg::ViewSpan<DevTs>...>& dev_spans,
                                    std::index_sequence<I, Is...>) {

    /* fill in pointers for the device -- we're relying on managed memory for this simple wrapper */
    typename std::tuple_element_t<I, typename ttg::View<HostT, DevTs...>::span_tuple_type>::element_type *ptr;
    size_t size;
    ptr = view.template get_device_ptr<I>();
    size = view.template get_device_size<I>();
    //cudaMalloc(&ptr, span.size_bytes());
    std::get<I>(dev_spans) = ttg::ViewSpan(ptr, size, view.template get_scope<I>());

    /* copy data to device */
    //cudaMemcpy(ptr, span.data(), span.size_bytes(), cudaMemcpyHostToDevice);
    if (view.template get_span<I>().is_sync_in()) {
#if defined(TTG_HAVE_CUDA) && defined(TTG_USE_CUDA_PREFETCH)
      cudaMemPrefetchAsync(span.data(), span.size_bytes(), 0, *ts_stream);
#endif // TTG_USE_CUDA_PREFETCH
    }

    if constexpr(sizeof...(Is) > 0) {
      create_view_on_device(view, dev_spans, std::index_sequence<Is...>{});
    }
  }

  template<typename HostT, typename... ViewSpanTs, std::size_t... Is>
  auto make_view_from_tuple(HostT& obj, std::tuple<ttg::ViewSpan<ViewSpanTs>...>& spans, std::index_sequence<Is...>) {
    return ttg::make_view(obj, std::get<Is>(spans)...);
  }

  template<typename... ViewTs, std::size_t I, std::size_t... Is>
  inline void create_on_device(std::tuple<ViewTs...>& views, std::tuple<ViewTs...>& dev_views, std::index_sequence<I, Is...>) {

    using view_tuple_t = typename std::tuple<ViewTs...>;
    auto& view = std::get<I>(views);
    typename std::tuple_element_t<I, view_tuple_t>::span_tuple_type dev_spans;
    create_view_on_device(view, dev_spans, std::make_index_sequence<std::tuple_element_t<I, view_tuple_t>::size()>());

    /* set the view for the device */
    std::get<I>(dev_views) = make_view_from_tuple(view.get_host_object(), dev_spans, std::make_index_sequence<std::tuple_size_v<decltype(dev_spans)>>{});
    if constexpr(sizeof...(Is) > 0) {
      create_on_device(views, dev_views, std::index_sequence<Is...>{});
    }
  }

  template<typename HostT, typename... DevTs, std::size_t I, std::size_t... Is>
  inline void sync_view_to_host(ttg::View<HostT, DevTs...>& dev_view, std::index_sequence<I, Is...>) {
    /* prefetch back to host */
    auto span = dev_view.template get_span<I>();

    /* prefetch data from device */
    if (span.is_sync_out()) {
#if defined(TTG_HAVE_CUDA) && defined(TTG_USE_CUDA_PREFETCH)
      cudaMemPrefetchAsync(span.data(), span.size_bytes(), cudaCpuDeviceId, *ts_stream);
#endif // TTG_USE_CUDA_PREFETCH
    }

    if constexpr(sizeof...(Is) > 0) {
      sync_view_to_host(dev_view, std::index_sequence<Is...>{});
    }
  }

  template<typename... ViewTs, std::size_t I, std::size_t... Is>
  inline void sync_back_to_host(std::tuple<ViewTs...>& dev_views, std::index_sequence<I, Is...>) {

    sync_view_to_host(std::get<I>(dev_views), std::make_index_sequence<std::tuple_element_t<I, std::tuple<ViewTs...>>::size()>());

    if constexpr(sizeof...(Is) > 0) {
      sync_back_to_host(dev_views, std::index_sequence<Is...>{});
    }
  }

  template<typename keyT,
           bool have_outterm_tuple,
           typename DevViewFuncT,
           typename DevKernelFuncT,
           typename DevOutFuncT,
           typename... input_edge_valuesT,
           typename... output_edgesT,
           typename... Args>
  auto make_device_tt_helper(DevViewFuncT &&view_func,
                             DevKernelFuncT &&kernel_func,
                             DevOutFuncT &&out_func,
                             ttg::ExecutionSpace space,
                             const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
                             const std::tuple<output_edgesT...> &outedges,
                             const std::string &name,
                             const std::vector<std::string> &innames,
                             const std::vector<std::string> &outnames,
                             const ttg::typelist<Args...>& full_input_args) {

    using output_terminals_type = typename ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

    auto taskfn = [=](const keyT& key, Args... args) mutable {

#ifdef TTG_HAVE_CUDA
      if (nullptr == ts_stream) {
        ts_stream = new cudaStream_t();
        cudaStreamCreate(ts_stream);
      }
#endif // TTG_HAVE_CUDA

      auto views = view_func(key, std::forward<Args>(args)...);
      using view_tuple_t = std::remove_reference_t<decltype(views)>;
      constexpr std::size_t view_tuple_size = std::tuple_size_v<view_tuple_t>;
      /* 1) allocate memory on device */
      auto device_views = views;
      /* 2) move data from views to device */
      if constexpr(std::tuple_size_v<view_tuple_t> > 0) {
        create_on_device(views, device_views, std::make_index_sequence<view_tuple_size>());
      }
      /* 3) call kernel function */
      detail::invoke_with_unpacked_views(kernel_func, key, device_views, std::make_index_sequence<view_tuple_size>());
      /* 4) move data back out into host objects */
      if constexpr(std::tuple_size_v<view_tuple_t> > 0) {
        sync_back_to_host(device_views, std::make_index_sequence<view_tuple_size>());
      }
  #ifdef TTG_HAVE_CUDA
      /* wait for the */
      cudaStreamSynchronize(*ts_stream);
  #endif // TTG_HAVE_CUDA
      /* 5) call output function */
      detail::invoke_out_with_unpacked_views(out_func, key, views, std::make_index_sequence<view_tuple_size>());
    };

  using wrapT = typename CallableWrapTTArgsAsTypelist<decltype(taskfn), void, have_outterm_tuple, keyT, output_terminals_type,
                                                      ttg::typelist<Args...>>::type;

  return std::make_unique<wrapT>(std::move(taskfn), inedges, outedges, name, innames, outnames);

  }


} // namespace detail


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

  using output_terminals_type = typename ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

  constexpr auto void_key = ttg::meta::is_void_v<keyT>;

  // gross list of candidate argument types
  using gross_candidate_func_args_t = ttg::meta::typelist<
      ttg::meta::candidate_argument_bindings_t<std::add_const_t<keyT>>,
      ttg::meta::candidate_argument_bindings_t<typename ttg::Edge<keyT, input_edge_valuesT>::value_type>...,
      ttg::meta::typelist<output_terminals_type &, void>>;

  // net list of candidate argument types excludes the empty typelists for void arguments
  using candidate_func_args_t = ttg::meta::filter_t<gross_candidate_func_args_t, ttg::meta::typelist_is_not_empty>;

  // gross argument typelist for invoking func, can include void for optional args
  constexpr static auto func_is_generic = ttg::meta::is_generic_callable_v<DevViewFuncT>;
  using gross_func_args_t = decltype(ttg::meta::compute_arg_binding_types_r<void>(view_func, candidate_func_args_t{}));
  constexpr auto DETECTED_HOW_TO_INVOKE_GENERIC_FUNC =
      func_is_generic ? !std::is_same_v<gross_func_args_t, ttg::typelist<>> : true;
  static_assert(DETECTED_HOW_TO_INVOKE_GENERIC_FUNC,
                "ttd::make_tt(func, inedges, ...): could not detect how to invoke generic callable func, either the "
                "signature of func "
                "is faulty, or inedges does match the expected list of types, or both");

  // net argument typelist
  using func_args_t = ttg::meta::drop_void_t<gross_func_args_t>;
  constexpr auto num_args = std::tuple_size_v<func_args_t>;

  // if given task id, make sure it's passed via const lvalue ref
  constexpr bool TASK_ID_PASSED_AS_CONST_LVALUE_REF =
      !void_key ? ttg::meta::probe_first_v<ttg::meta::is_const_lvalue_reference, true, func_args_t> : true;
  static_assert(TASK_ID_PASSED_AS_CONST_LVALUE_REF,
                "ttg::make_tt(func, ...): if given to func, the task id must be passed by const lvalue ref");

  // if given out-terminal tuple, make sure it's passed via nonconst lvalue ref
  constexpr bool have_outterm_tuple =
      func_is_generic ? !ttg::meta::is_last_void_v<gross_func_args_t>
                      : ttg::meta::probe_last_v<ttg::meta::decays_to_output_terminal_tuple, false, gross_func_args_t>;
  constexpr bool OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF =
      have_outterm_tuple ? ttg::meta::probe_last_v<ttg::meta::is_nonconst_lvalue_reference, false, func_args_t> : true;
  static_assert(
      OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF,
      "ttg::make_tt(func, ...): if given to func, the output terminal tuple must be passed by nonconst lvalue ref");

  // TT needs actual types of arguments to func ... extract them and pass to CallableWrapTTArgs
  using input_edge_value_types = ttg::meta::typelist<std::decay_t<input_edge_valuesT>...>;
  // input_args_t = {input_valuesT&&...}
  using input_args_t = typename ttg::meta::take_first_n<
      typename ttg::meta::drop_first_n<func_args_t, std::size_t(void_key ? 0 : 1)>::type,
      std::tuple_size_v<func_args_t> - (void_key ? 0 : 1) - (have_outterm_tuple ? 1 : 0)>::type;
  constexpr auto NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF =
      !ttg::meta::is_any_nonconst_lvalue_reference_v<input_args_t>;
  static_assert(
      NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF,
      "ttg::make_tt(func, inedges, outedges): one or more arguments to func can only be passed by nonconst lvalue "
      "ref; this is illegal, should only pass arguments as const lavlue ref or (nonconst) rvalue ref");
  using decayed_input_args_t = ttg::meta::decayed_typelist_t<input_args_t>;
  // 3. full_input_args_t = edge-types with non-void types replaced by input_args_t
  using full_input_args_t = ttg::meta::replace_nonvoid_t<input_edge_value_types, input_args_t>;

  return detail::make_device_tt_helper<keyT, have_outterm_tuple>(std::forward<DevViewFuncT>(view_func),
                                                                 std::forward<DevKernelFuncT>(kernel_func),
                                                                 std::forward<DevOutFuncT>(out_func),
                                                                 space, inedges, outedges, name, innames, outnames,
                                                                 full_input_args_t{});
}

#if 0
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

  /* drop the host function */
  return make_device_tt(view_func, kernel_func, out_func, space, inedges, outedges, name, innames, outnames);
}
#endif // 0
#endif // TTG_MAKE_DEVICE_TT_H

// clang-format on
