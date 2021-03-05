# Managing Continuous Intergration (CI) {#CI-Administration-Notes}

## Basic Facts
* TTG uses GitHub Actions (GHA) for its CI service
* GHA CI configuration is in file `.github/workflows/cmake.yml`. Only Linux and MacOS builds are currently supported.
* Unlike earlier CI setups, there is no need to cache TTG prerequisites; default system-wide packages are used for most prerequisites, and the rest is compiled from source every time. 
* Doxygen documentation deployment uses a Github token that is defined as variable `GH_TTG_TOKEN` in GHA's TTG repo settings' [secrets](https://github.com/TESSEorg/ttg/settings/secrets/actions).

# Debugging GitHub Actions jobs

## Local debugging

GHA Linux jobs run on stock 20.04 Ubuntu, thus they can be reproduced easily in, e.g., a stock Ubuntu container. MacOS jobs run on (x86) MacOS virtual machines, thus require a local Mac to troubleshoot.
