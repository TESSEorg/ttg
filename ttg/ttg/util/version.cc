// SPDX-License-Identifier: BSD-3-Clause
#include "ttg/util/version.h"

namespace ttg {
  std::array<int, 3> version() { return {TTG_MAJOR_VERSION, TTG_MINOR_VERSION, TTG_MICRO_VERSION}; }

  const char* git_revision() noexcept {
    static const char revision[] = TTG_GIT_REVISION;
    return revision;
  }

  const char* git_description() noexcept {
    static const char description[] = TTG_GIT_DESCRIPTION;
    return description;
  }

}  // namespace ttg
