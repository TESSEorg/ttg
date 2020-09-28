# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# This is based on:
#   https://github.com/vector-of-bool/CMakeCM/blob/master/modules/FindFilesystem.cmake

#[=======================================================================[.rst:

FindCXXStdExecution
##############

This module supports the C++17 standard library's execution utilities. Use the
:imp-target:`std::execution` imported target to

Imported Targets
****************

.. imp-target:: std::execution

    The ``std::execution`` imported target is defined when any requested
    version of the C++ execution library component has been found.

    If no version of the execution library is available, this target will not
    be defined.

    .. note::
        This target has ``cxx_std_17`` as an ``INTERFACE``
        :ref:`compile language standard feature <req-lang-standards>`. Linking
        to this target will automatically enable C++17 if no later standard
        version is already required on the linking target.


.. _fs.variables:

Variables
*********

.. variable:: CXX_HAVE_EXECUTION_HEADER

    Set to ``TRUE`` when usable <execution> header is found.

Examples
********

Using `find_package(CXXStdExecution)` with no component arguments:

.. code-block:: cmake

    find_package(CXXStdExecution REQUIRED)

    add_executable(my-program main.cpp)
    target_link_libraries(my-program PRIVATE std::execution)


#]=======================================================================]


if(TARGET std::execution)
  # This module has already been processed. Don't do it again.
  return()
endif()

include(CMakePushCheckState)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)

cmake_push_check_state()

set(CMAKE_REQUIRED_QUIET ${CXXStdExecution_FIND_QUIETLY})

# All of our tests required C++17 or later
set(CMAKE_CXX_STANDARD 17)

set(_found FALSE)

# We have execution header, but how do we use it? Do link checks
string(CONFIGURE [[
  #include <algorithm>
  #include <vector>
  #include <execution>
  int main(int argc, char** argv) {
    std::vector<int> v{0,1,2};
    std::for_each(std::execution::par_unseq, begin(v), end(v),
                  [](auto&& i) {i *= 2;});
    return 0;
  }
  ]] code @ONLY)

# Try to compile a simple filesystem program without any linker flags
check_cxx_source_compiles("${code}" CXX_EXECUTION_NO_LINK_NEEDED)

set(can_link ${CXX_EXECUTION_NO_LINK_NEEDED})

if(NOT CXX_EXECUTION_NO_LINK_NEEDED)
  if (NOT TBB_FOUND)
    find_package(TBB)
    if (TBB_FOUND)
      # set up an interface library for TBB a la https://github.com/justusc/FindTBB
      include(ImportTBB)
      import_tbb()
    endif()
  endif()
  if (TARGET tbb)
    set(prev_libraries ${CMAKE_REQUIRED_LIBRARIES})
    # Try to link a simple program with the ${TBB_LIBRARIES}
    set(CMAKE_REQUIRED_LIBRARIES ${prev_libraries} tbb)
    check_cxx_source_compiles("${code}" CXX_EXECUTION_TBB_NEEDED)
    set(can_link ${CXX_EXECUTION_TBB_NEEDED})
  endif()
endif()

if(can_link)
  add_library(std::execution INTERFACE IMPORTED)
  target_compile_features(std::execution INTERFACE cxx_std_17)
  set(_found TRUE)

  if(CXX_EXECUTION_NO_LINK_NEEDED)
    # Nothing to add...
  elseif(CXX_EXECUTION_TBB_NEEDED)
    target_link_libraries(std::execution INTERFACE tbb)
  endif()
endif()

cmake_pop_check_state()

if(NOT TARGET std::execution AND CXXStdFilesystem_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot compile and link programs that #include <execution>")
endif()
