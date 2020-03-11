# Developer Guide {#devguide}

## Keeping the subprojects in sync

TTG uses a set of subprojects as dependencies: MADNESS, PaRSEC and
Boost at least. In order to keep TTG in sync with each subproject, one
needs to update the following files:

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


