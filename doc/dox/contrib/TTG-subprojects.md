# Subprojects in TTG {#TTG-subprojects}

TTG uses a set of subprojects as dependencies: MADNESS, PaRSEC and
Boost at least. These projects are pulled from git repositories in
<BUILDDIR>/external/<SUBPROJECT>-src/ directories, stamp files (for
Makefile) are created in <BUILDDIR>/external/<SUBPROJECT>-download
directories, and they are compiled either in their source directory,
or in <BUILD_DIR>/external/<SUBPROJECT>-build directories.

## Keeping the subprojects in sync

In order to keep TTG in sync with each subproject, one needs to update the following files:

1. cmake/modules/FindOrFetch*.cmake : these files decide what is the
source of the git repository for each project

2. cmake/modules/ExternalDependenciesVersions.cmake: define, for
each subproject the version / TAG to use. For Boost, a VERSION is
required, for MADNESS and PaRSEC, the TAG variable can store either
a named TAG or a commit identifier.

The build system does not always update the subproject source when
either of these files are changed. It is recommended to remove the
build/external/ directory if versioning errors are observed during the
build process.


