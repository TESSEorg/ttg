# Managing GitHub Actions {#GitHUb-Actions-Administration-Notes}

## Basic Facts

* GitHub Actions (GHA) configuration is in file `.github/workflows/cmake.yml`. Linux and MacOS builds are currently
  supported.
* No built prerequisites are cached, hence MADNESS, PaRSEC, and BTAS (and their dependencies) are built from source
  every time. Although `ccache` is used, as of now it does not appear that `ccache`'s _cache_ is actually used.
* Doxygen deployment script uses Github token that is defined as variable `GH_TTG_TOKEN` in GHA's TTG repo settings.

# Debugging GitHub Actions jobs

No local debugging is possible yet.
