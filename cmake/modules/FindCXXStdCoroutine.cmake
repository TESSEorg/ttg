# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# This is copied from:
#   https://github.com/vector-of-bool/CMakeCM/blob/master/modules/FindFilesystem.cmake

#[=======================================================================[.rst:

FindCXXStdCoroutine
##############

This module supports the C++20 standard library's coroutine utilities. Link your target to the
:imp-target:`std::coroutine` imported target to provide standard C++ coroutine API.

Options
*******

The ``COMPONENTS`` argument to this module supports the following values:

.. find-component:: Experimental
    :name: coro.Experimental

    Allows the module to find the "experimental" version of the
    Coroutine library. This is the library that should be used with the
    ``std::experimental::coroutine`` namespace.

.. find-component:: Final
    :name: coro.Final

    Finds the final C++20 standard version of the coroutine library.

If no components are provided, behaves as if the
:find-component:`coro.Final` component was specified.

If both :find-component:`coro.Experimental` and :find-component:`coro.Final` are
provided, first looks for ``Final``, and falls back to ``Experimental`` in case
of failure. If ``Final`` is found, :imp-target:`std::coroutine` and all
:ref:`variables <coro.variables>` will refer to the ``Final`` version.


Imported Targets
****************

.. imp-target:: std::coroutine

    The ``std::coroutine`` imported target is defined when any requested
    version of the C++ coroutine library has been found, whether it is
    *Experimental* or *Final*.

    If no version of the coroutine library is available, this target will not
    be defined.

    .. note::
        This target has ``cxx_std_20`` as an ``INTERFACE``
        :ref:`compile language standard feature <req-lang-standards>`. Linking
        to this target will automatically enable C++20 if no later standard
        version is already required on the linking target.


.. coro.variables:

Variables
*********

.. variable:: CXX_COROUTINE_COMPONENT

    Set to ``Final`` when the :find-component:`coro.Final` version of C++
    coroutine library was found, ``Experimental`` when
    the :find-component:`coro.Experimental` version of C++
    coroutine library was found, otherwise not defined.

.. variable:: CXX_COROUTINE_HAVE_CORO

    Set to ``TRUE`` when a coroutine header was found.

.. variable:: CXX_COROUTINE_HEADER

    Set to either ``coroutine`` or ``experimental/coroutine`` depending on
    whether :find-component:`coro.Final` or :find-component:`coro.Experimental` was
    found.

.. variable:: CXX_COROUTINE_NAMESPACE

    Set to either ``std::coroutine`` or ``std::experimental::coroutine``
    depending on whether :find-component:`coro.Final` or
    :find-component:`coro.Experimental` was found.


Examples
********

Using `find_package(Coroutine)` with no component arguments:

.. code-block:: cmake

    find_package(Coroutine REQUIRED)

    add_executable(my-program main.cpp)
    target_link_libraries(my-program PRIVATE std::coroutine)


#]=======================================================================]


if(TARGET std::coroutine)
  # This module has already been processed. Don't do it again.
  return()
endif()

include(CMakePushCheckState)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)

cmake_push_check_state()

set(CMAKE_REQUIRED_QUIET ${CXXStdCoroutine_FIND_QUIETLY})

# Normalize and check the component list we were given
set(CXXStdCoroutines_want_components ${CXXStdCoroutine_FIND_COMPONENTS})
if(CXXStdCoroutine_FIND_COMPONENTS STREQUAL "")
  set(CXXStdCoroutines_want_components Final)
endif()

# Warn on any unrecognized components
set(CXXStdCoroutines_extra_components ${CXXStdCoroutines_want_components})
list(REMOVE_ITEM CXXStdCoroutines_extra_components Final Experimental)
foreach(component IN LISTS CXXStdCoroutines_extra_components)
  message(WARNING "Extraneous find_package component for CXXStdCoroutine: ${component}")
endforeach()

# clang may need to use -stdlib=c++ to have coroutines
# gcc/libstdc++ needs -fcoroutines
set(CXXStdCoroutines_find_options "" "-stdlib=libc++" "-fcoroutines")
set(CXXStdCoroutines_want_components_ordered "${CXXStdCoroutines_want_components}")
list(SORT CXXStdCoroutines_want_components_ordered ORDER DESCENDING)  # Final before Experimental

foreach(component IN LISTS CXXStdCoroutines_want_components_ordered)
  if(component STREQUAL "Final")
    set(_coro_header coroutine)
    set(_coro_namespace std)
  else()
    set(_coro_header experimental/coroutine)
    set(_coro_namespace std::experimental)
  endif()
  foreach(option IN LISTS CXXStdCoroutines_find_options)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_FLAGS "${option}")
    check_include_file_cxx("${_coro_header}" _CXX_COROUTINE_HAVE_HEADER)
    mark_as_advanced(_CXX_COROUTINE_HAVE_HEADER)
    cmake_pop_check_state()
    if(_CXX_COROUTINE_HAVE_HEADER)
      set(CXXStdCoroutines_option "${option}" CACHE STRING "The compiler option needed to use <coroutine>")
      mark_as_advanced(CXXStdCoroutines_option)
      set(CXX_COROUTINE_COMPONENT "${component}" CACHE STRING "The component of CXXStdCoroutine package found")
      # We found a header, skip rest of tests
      break()
    else()
      unset(_CXX_COROUTINE_HAVE_HEADER CACHE)
    endif()
  endforeach()  # options
  if (_CXX_COROUTINE_HAVE_HEADER)
    break()
  endif()
endforeach() # components

set(CXX_COROUTINE_HAVE_CORO ${_CXX_COROUTINE_HAVE_HEADER} CACHE BOOL "TRUE if we have the C++ coroutine headers")
set(CXX_COROUTINE_HEADER ${_coro_header} CACHE STRING "The header that should be included to obtain the coroutine APIs")
set(CXX_COROUTINE_NAMESPACE ${_coro_namespace} CACHE STRING "The C++ namespace that contains the coroutine APIs")

set(_found FALSE)

if(CXX_COROUTINE_HAVE_CORO)
  # We have some coroutine library available. Do compile checks
  string(CONFIGURE [[
        #include <@CXX_COROUTINE_HEADER@>

        int main() {
            auto x = @CXX_COROUTINE_NAMESPACE@::suspend_always{};
            return 0;
        }
    ]] code @ONLY)

  cmake_push_check_state()
  if (DEFINED CXXStdCoroutines_option)
    list(APPEND CMAKE_REQUIRED_FLAGS "${CXXStdCoroutines_option}")
  endif()
  check_cxx_source_compiles("${code}" CXX_COROUTINE_NO_LINK_NEEDED)
  cmake_pop_check_state()

  if(CXX_COROUTINE_NO_LINK_NEEDED)
    add_library(std::coroutine INTERFACE IMPORTED GLOBAL)
    target_compile_features(std::coroutine INTERFACE cxx_std_20)
    if (DEFINED CXXStdCoroutines_option)
      target_compile_options(std::coroutine INTERFACE "${CXXStdCoroutines_option}")
    endif()
    set(_found TRUE)
  endif()
endif()

cmake_pop_check_state()

set(CXXStdCoroutine_FOUND ${_found} CACHE BOOL "TRUE if we can compile and link a program using std::coroutine" FORCE)

if(CXXStdCoroutine_FIND_REQUIRED AND NOT TARGET std::coroutine)
  message(FATAL_ERROR "Cannot Compile simple program using std::coroutine")
endif()
