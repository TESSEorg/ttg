# This file should be sourced into the shell environment

# Show the executed command, but don't affect spawned shells
trap 'echo "# $BASH_COMMAND"' DEBUG # Show commands

if [[ -z "$SPACK_SETUP" || ! -e "$SPACK_SETUP" ]]; then
   echo Error! Environment variable \$SPACK_SETUP must point
   echo to a valid setup-env.sh Spack setup script.
   exit 1
fi
source $SPACK_SETUP
spack env activate -V parsec
spack load cmake openblas ninja openmpi

export CUDA=OFF
export HIP=OFF
export LEVEL_ZERO=OFF
if [ "$DEVICE" = "gpu_nvidia" ]; then
   spack load cuda
   CUDA=ON
elif [ "$DEVICE" = "gpu_amd" ]; then
   HIP=ON
elif [ "$DEVICE" = "gpu_intel" ]; then
   LEVEL_ZERO=ON
fi

