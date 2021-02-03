# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_MADNESS_TAG b22ee85059e6ccc9a6e803ba0550652ece8d9df1)
set(TTG_TRACKED_PARSEC_TAG 84ff0d7141d0c6548b60567d12c301bb94232473)
set(TTG_TRACKED_BTAS_TAG 1c6099ed2d709896430a892b05bcb94b306f76c9)