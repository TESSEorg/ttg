# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TTG_TRACKED_VG_CMAKE_KIT_TAG 092efee765e039b02e0a9aaf013c12fc3c4e89cf)  # used to provide "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_MADNESS_TAG 2eb3bcf0138127ee2dbc651f1aabd3e9b0def4e3)
set(TTG_TRACKED_PARSEC_TAG 58f8f3089ecad2e8ee50e80a9586e05ce8873b1c)
set(TTG_TRACKED_BTAS_TAG 4e8f5233aa7881dccdfcc37ce07128833926d3c2)
set(TTG_TRACKED_TILEDARRAY_TAG 493c109379a1b64ddd5ef59f7e33b95633b68d73)

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_OLDEST_BOOST_VERSION 1.66)
