# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_VG_CMAKE_KIT_TAG 7ea2d4d3f8854b9e417f297fd74d6fc49aa13fd5)  # used to provide "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 8788aea9758bfe6479cc23d39e6c77b7528009db)
set(TTG_TRACKED_PARSEC_TAG 25d1931e863b6741e453112d2117d85ad32e7fba)
set(TTG_TRACKED_BTAS_TAG 4e8f5233aa7881dccdfcc37ce07128833926d3c2)
set(TTG_TRACKED_TILEDARRAY_TAG 493c109379a1b64ddd5ef59f7e33b95633b68d73)
