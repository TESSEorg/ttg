These notes describe how to build a stand-alone TTG image useful for experimentation and/or provisioning computational results (e.g. for creating supplementary info for a journal article).

# Docker container notes
These notes assume that Docker is installed on your machine and that you start at the top of the TTG source tree.

## Create/build TTG Docker image
1. Create a Docker image: `bin/docker-build.sh`
2. Run a container using the newly created
   image: `docker run --privileged -i -t --rm ttg-dev:latest /sbin/my_init -- bash -l`
3. To run tests/examples: `cmake --build /home/tesse/ttg-build --target check`

## Notes
- Important locations:
  - source: `/home/tesse/ttg`
  - build: `/home/tesse/ttg-build`
  - install: `/home/tesse/ttg-install`
