#ifndef TTG_PARSEC_DEVICEFUNC_H
#define TTG_PARSEC_DEVICEFUNC_H

#if defined(TTG_HAVE_CUDART)
#include <cuda.h>
#endif

#include "ttg/parsec/task.h"
#include <parsec/mca/device/device_gpu.h>

#if defined(PARSEC_HAVE_CUDA)
#include <parsec/mca/device/cuda/device_cuda.h>
#endif // PARSEC_HAVE_CUDA

namespace ttg_parsec {
  namespace detail {
    template<typename... Views, std::size_t I, std::size_t... Is>
    inline bool register_device_memory(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {
      static_assert(I < MAX_PARAM_COUNT,
                    "PaRSEC only supports MAX_PARAM_COUNT device input/outputs. "
                    "Increase MAX_PARAM_COUNT and recompile PaRSEC/TTG.");
      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;
      auto& view = std::get<I>(views);
      bool is_current = false;
      static_assert(ttg::is_buffer_v<view_type> || ttg_parsec::is_devicescratch_v<view_type>);
      /* get_parsec_data is overloaded for buffer and devicescratch */
      parsec_data_t* data = detail::get_parsec_data(view);
      /* TODO: check whether the device is current */

      auto flow_flags = PARSEC_FLOW_ACCESS_RW;
      bool pushout = false;
      if constexpr (std::is_const_v<view_type>) {
        flow_flags = PARSEC_FLOW_ACCESS_READ;
      } else if constexpr (ttg_parsec::is_devicescratch_v<view_type>) {
        switch(view.scope()) {
          case ttg::scope::Allocate:
            flow_flags = PARSEC_FLOW_ACCESS_NONE;
            break;
          case ttg::scope::SyncIn:
            flow_flags = PARSEC_FLOW_ACCESS_READ;
            break;
        }
      }
      assert(nullptr != detail::parsec_ttg_caller->dev_ptr);
      parsec_gpu_task_t *gpu_task = detail::parsec_ttg_caller->dev_ptr->gpu_task;
      parsec_flow_t *flows = detail::parsec_ttg_caller->dev_ptr->flows;

      std::cout << "register_device_memory task " << detail::parsec_ttg_caller << " data " << I << " "
                << data << " size " << data->nb_elts << std::endl;

      /* build the flow */
      /* TODO: reuse the flows of the task class? How can we control the sync direction then? */
      flows[I] = parsec_flow_t{.name = nullptr,
                               .sym_type = PARSEC_SYM_INOUT,
                               .flow_flags = static_cast<uint8_t>(flow_flags),
                               .flow_index = I,
                               .flow_datatype_mask = ~0 };

      gpu_task->flow_nb_elts[I] = data->nb_elts; // size in bytes
      gpu_task->flow[I] = &flows[I];

      if (pushout) {
        std::cout << "PUSHOUT " << I << std::endl;
        gpu_task->pushout |= 1<<I;
      }

      /* set the input data copy, parsec will take care of the transfer
       * and the buffer will look at the parsec_data_t for the current pointer */
      //detail::parsec_ttg_caller->parsec_task.data[I].data_in = data->device_copies[data->owner_device];
      detail::parsec_ttg_caller->parsec_task.data[I].data_in = data->device_copies[0];
      detail::parsec_ttg_caller->parsec_task.data[I].source_repo_entry = NULL;

      if constexpr (sizeof...(Is) > 0) {
        is_current |= register_device_memory(views, std::index_sequence<Is...>{});
      }
      return is_current;
    }
  } // namespace detail

  /* Takes a tuple of ttg::Views or ttg::buffers and register them
   * with the currently executing task. Returns true if all memory
   * is current on the target device, false if transfers are required. */
  template<typename... Views>
  inline bool register_device_memory(std::tuple<Views&...> &views) {
    if (nullptr == detail::parsec_ttg_caller) {
      throw std::runtime_error("register_device_memory may only be invoked from inside a task!");
    }

    if (nullptr == detail::parsec_ttg_caller->dev_ptr) {
      throw std::runtime_error("register_device_memory called inside a non-gpu task!");
    }

    bool is_current = detail::register_device_memory(views, std::index_sequence_for<Views...>{});

    /* reset all entries in the current task */
    for (int i = sizeof...(Views); i < MAX_PARAM_COUNT; ++i) {
      detail::parsec_ttg_caller->parsec_task.data[i].data_in = nullptr;
      detail::parsec_ttg_caller->dev_ptr->flows[i].flow_flags = PARSEC_FLOW_ACCESS_NONE;
      detail::parsec_ttg_caller->dev_ptr->flows[i].flow_index = i;
      detail::parsec_ttg_caller->dev_ptr->gpu_task->flow[i] = &detail::parsec_ttg_caller->dev_ptr->flows[i];
      detail::parsec_ttg_caller->dev_ptr->gpu_task->flow_nb_elts[i] = 0;
    }

    return is_current;
  }

  namespace detail {
    template<typename... Views, std::size_t I, std::size_t... Is, bool DeviceAvail = false>
    inline void mark_device_out(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {

      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;
      auto& view = std::get<I>(views);

      /* get_parsec_data is overloaded for buffer and devicescratch */
      parsec_data_t* data = detail::get_parsec_data(view);
      /* find the data copy and mark it as pushout */
      int i = 0;
      parsec_gpu_task_t *gpu_task = detail::parsec_ttg_caller->dev_ptr->gpu_task;
      parsec_gpu_exec_stream_t *stream = detail::parsec_ttg_caller->dev_ptr->stream;
      /* enqueue the transfer into the compute stream to come back once the compute and transfer are complete */

#if defined(TTG_HAVE_CUDART) && defined(PARSEC_HAVE_CUDA)
      parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)stream;
      cudaMemcpyAsync(data->device_copies[0]->device_private,
                      data->device_copies[data->owner_device]->device_private,
                      data->nb_elts, cudaMemcpyDeviceToHost, cuda_stream->cuda_stream);
#else
      static_assert(DeviceAvail, "No device implementation detected!");
#endif // defined(PARSEC_HAVE_CUDA)

#if 0
      while (detail::parsec_ttg_caller->parsec_task.data[i].data_in != nullptr) {
        if (detail::parsec_ttg_caller->parsec_task.data[i].data_in == data->device_copies[0]) {
          gpu_task->pushout |= 1<<i;
          break;
        }
        ++i;
      }
#endif // 0
      if constexpr (sizeof...(Is) > 0) {
        // recursion
        mark_device_out(views, std::index_sequence<Is...>{});
      }
    }
  } // namespace detail

  template<typename... Buffer>
  inline void mark_device_out(std::tuple<Buffer&...> &b) {

    if (nullptr == detail::parsec_ttg_caller) {
      throw std::runtime_error("mark_device_out may only be invoked from inside a task!");
    }

    if (nullptr == detail::parsec_ttg_caller->dev_ptr) {
      throw std::runtime_error("mark_device_out called inside a non-gpu task!");
    }

    detail::mark_device_out(b, std::index_sequence_for<Buffer...>{});
  }

  namespace detail {

    template<typename... Views, std::size_t I, std::size_t... Is>
    inline void post_device_out(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {

      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;

      if constexpr (!std::is_const_v<view_type>) {
        auto& view = std::get<I>(views);

        /* get_parsec_data is overloaded for buffer and devicescratch */
        parsec_data_t* data = detail::get_parsec_data(view);

        data->device_copies[0]->version++;
        data->owner_device = 0;
      }

      if constexpr (sizeof...(Is) > 0) {
        // recursion
        post_device_out(views, std::index_sequence<Is...>{});
      }
    }
  } // namespace detail
  template<typename... Buffer>
  inline void post_device_out(std::tuple<Buffer&...> &b) {
    detail::post_device_out(b, std::index_sequence_for<Buffer...>{});
  }


} // namespace ttg_parsec

#endif // TTG_PARSEC_DEVICEFUNC_H