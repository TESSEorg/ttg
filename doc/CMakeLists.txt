# SPDX-License-Identifier: BSD-3-Clause
# this will depend on all worker targets
add_custom_target_subproject(ttg doc)

#-- include doxygen code
add_subdirectory(dox)
