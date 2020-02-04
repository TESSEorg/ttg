# Managing Travis Builds {#Travis-CI-Administration-Notes}

## Basic Facts
* Travis CI configuration is in file `.travis.yml`, and build scripts are in `bin/build-*linux.sh`. Only Linux builds are currently supported.
* Installation directories for all prerequisites (MPICH, etc.) are _cached_. **Build scripts only verify the presence of installed directories, and do not update them if their configuration (e.g. static vs. shared, or code version) has changed. _Thus it is admin's responsibility to manually wipe out the cache on a per-branch basis_.** It is the easiest to do via the Travis-CI web interface (click on 'More Options' menu at the top right, select 'Caches', etc.).
* Doxygen deployment script uses Github token that is defined as variable  `GH_TTG_TOKEN` in Travis-CI's TTG repo settings.

# Debugging Travis-CI jobs

## Local debugging

Follow the instructions contained in [docker-travis.md](https://github.com/TESSEorg/ttg/blob/master/bin/docker-travis.md) .
