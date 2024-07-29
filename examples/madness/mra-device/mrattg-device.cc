#include <ttg.h>
#include "tensor.h"
#include "tensorview.h"
#include "functionnode.h"
#include "functiondata.h"
#include "kernels.h"
#include "gaussian.h"
#include "functionfunctor.h"
#include "../../mrakey.h"

#if 0
/// Project the scaling coefficients using screening and test norm of difference coeffs.  Return true if difference coeffs negligible.
template <typename FnT, typename T, mra::Dimension NDIM>
static bool fcoeffs(
  const ttg::Buffer<FnT>& f,
  const mra::FunctionData<T, NDIM>& functiondata,
  const mra::Key<NDIM>& key,
  const T thresh,
  mra::Tensor<T,NDIM>& coeffs)
{
  bool status;

  if (mra::is_negligible(*f.host_ptr(),mra::Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
    coeffs = 0.0;
    status = true;
  }
  else {

    /* global function data */
    // TODO: need to make our own FunctionData with dynamic K
    const auto& phibar = functiondata.get_phibar();
    const auto& hgT = functiondata.get_hgT();

    const std::size_t K = coeffs.dim(0);

    /* temporaries */
    bool is_leaf;
    auto is_leaf_scratch = ttg::make_scratch(&is_leaf, ttg::scope::Allocate);
    const std::size_t tmp_size = project_tmp_size<NDIM>(K);
    T* tmp = new T[tmp_size]; // TODO: move this into make_scratch()
    auto tmp_scratch = ttg::make_scratch(tmp, ttg::scope::Allocate, tmp_size);

    /* TODO: cannot do this from a function, need to move it into the main task */
    co_await ttg::device::select(f, coeffs.buffer(), phibar.buffer(), hgT.buffer(), tmp, is_leaf_scratch);
    auto coeffs_view = coeffs.current_view();
    auto phibar_view = phibar.current_view();
    auto hgT_view    = hgT.current_view();
    T* tmp_device = tmp_scratch.device_ptr();
    bool *is_leaf_device = is_leaf_scratch.device_ptr();
    FnT* f_ptr = f.current_device_ptr();

    /* submit the kernel */
    submit_fcoeffs_kernel(f_ptr, key, coeffs_view, phibar_view, hgT_view, tmp_device,
                          is_leaf_device, ttg::device::current_stream());

    /* wait and get is_leaf back */
    co_await ttg::device::wait(is_leaf_scratch);
    status = is_leaf;
    /* todo: is this safe? */
    delete[] tmp;
  }
  co_return status;
}
#endif // 0

template<typename FnT, typename T, mra::Dimension NDIM>
auto make_project(
  ttg::Buffer<FnT>& f,
  const mra::FunctionData<T, NDIM>& functiondata,
  const T thresh, /// should be scalar value not complex
  ttg::Edge<mra::Key<NDIM>, void> control,
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, NDIM>> result)
{

  auto fn = [&](const mra::Key<NDIM>& key) -> ttg::device::Task {
    using tensor_type = typename mra::Tensor<T, NDIM>;
    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionReconstructedNode<T, NDIM>;
    node_type result;
    tensor_type& coeffs = result.coeffs;

    if (key.level() < initial_level(f)) {
      std::vector<mra::Key<NDIM>> bcast_keys;
      /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                We need to fix broadcast to support any ranges */
      for (auto child : children(key)) bcast_keys.push_back(child);
      ttg::broadcastk<0>(bcast_keys);
      coeffs.current_view() = T(1e7); // set to obviously bad value to detect incorrect use
      result.is_leaf = false;
    }
    else if (mra::is_negligible<FnT,T,NDIM>(*f.host_ptr(), mra::Domain<NDIM>:: template bounding_box<T>(key), mra::truncate_tol(key,thresh))) {
      /* zero coeffs */
      coeffs.current_view() = T(0.0);
      result.is_leaf = true;
    }
    else {
      /* here we actually compute: first select a device */
      //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
      /**
       * BEGIN FCOEFFS HERE
       * TODO: figure out a way to outline this into a function or coroutine
       */

      /* global function data */
      // TODO: need to make our own FunctionData with dynamic K
      const auto& phibar = functiondata.get_phibar();
      const auto& hgT = functiondata.get_hgT();

      const std::size_t K = coeffs.dim(0);

      /* temporaries */
      bool is_leaf;
      auto is_leaf_scratch = ttg::make_scratch(&is_leaf, ttg::scope::Allocate);
      const std::size_t tmp_size = project_tmp_size<NDIM>(K);
      T* tmp = new T[tmp_size]; // TODO: move this into make_scratch()
      auto tmp_scratch = ttg::make_scratch(tmp, ttg::scope::Allocate, tmp_size);

      /* TODO: cannot do this from a function, need to move it into the main task */
      co_await ttg::device::select(f, coeffs.buffer(), phibar.buffer(),
                                   hgT.buffer(), tmp_scratch, is_leaf_scratch);
      auto coeffs_view = coeffs.current_view();
      auto phibar_view = phibar.current_view();
      auto hgT_view    = hgT.current_view();
      T* tmp_device = tmp_scratch.device_ptr();
      bool *is_leaf_device = is_leaf_scratch.device_ptr();
      FnT* f_ptr = f.current_device_ptr();

      /* submit the kernel */
      submit_fcoeffs_kernel(f_ptr, key, coeffs_view, phibar_view, hgT_view, tmp_device,
                            is_leaf_device, ttg::device::current_stream());

      /* wait and get is_leaf back */
      co_await ttg::device::wait(is_leaf_scratch);
      result.is_leaf = is_leaf;
      /* todo: is this safe? */
      delete[] tmp;
      /**
       * END FCOEFFS HERE
       */

      if (!result.is_leaf) {
        std::vector<mra::Key<NDIM>> bcast_keys;
        for (auto child : children(key)) bcast_keys.push_back(child);
        ttg::broadcastk<0>(bcast_keys);
      }
    }
    ttg::send<1>(key, std::move(result)); // always produce a result
  };

  return ttg::make_tt(std::move(fn), ttg::edges(control), ttg::edges(result));
}

template<typename T, mra::Dimension NDIM>
void test(std::size_t K) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  mra::Domain<NDIM>::set_cube(-6.0,6.0);

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, NDIM>> project_result;

  // define a Gaussian
  auto gaussian = mra::Gaussian<T, NDIM>(T(3.0), {T(0.0),T(0.0),T(0.0)});
  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(&gaussian);
  auto project = make_project(gauss_buffer, functiondata, T(1e-6), project_control, project_result);
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);

  test<double, 3>(10);

  ttg::finalize();
}