# SPDX-License-Identifier: BSD-3-Clause

# {BLAS,LAPACK}_LIBRARIES may include IMPORTED targets such as OpenMP::OpenMP_C
# these targets are usually exist when these variables are "discovered", but they may not if they are already in CACHE

include(CMakeFindDependencyMacro)
include(CheckLanguage)

function(SanitizeLinalgLibraries)
    if (BLAS_LIBRARIES OR LAPACK_LIBRARIES)
        # look for OpenMP
        foreach (lang IN ITEMS C CXX Fortran)
            if (BLAS_LIBRARIES MATCHES OpenMP::OpenMP_${lang} OR LAPACK_LIBRARIES MATCHES OpenMP::OpenMP_${lang})
                check_language(${lang})
                if(CMAKE_${lang}_COMPILER)
                    enable_language(${lang})
                    find_dependency(OpenMP)
                else()
                    message(FATAL_ERROR "No support for LANGUAGE ${lang} but {BLAS,LAPACK}_LIBRARIES contains IMPORTED target OpenMP::OpenMP_${lang} that requires it")
                endif()
            endif()
        endforeach()
    endif()
endfunction()