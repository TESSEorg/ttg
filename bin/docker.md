These notes describe how to build a stand-alone TTG image useful for experimentation and/or provisioning computational results (e.g. for creating supplementary info for a journal article). If you want to use Docker to run/debug Travis-CI jobs, see [docker-travis.md](docker-travis.md)

# Docker container notes
These notes assume that Docker is installed on your machine and that you start at the top of the TTG source tree.

## Create/build Docker Travis image
1. Create a Docker image: `bin/docker-build.sh`
2. Run a container using the newly created image: `docker run --privileged -i -t --rm ttg-dev:latest /sbin/my_init -- bash -l`

## Notes
- Important locations:
  - source: `/home/tesse/ttg`
  - build: `/home/tesse/ttg-build`
  - install: `/home/tesse/ttg-install`
