# eri-pch — pivoted Cholesky on 4-center AO ERIs

Integral-direct pivoted Cholesky decomposition of the 4-center AO
electron-repulsion integral matrix, following Koch *et al.*,
*J. Chem. Phys.* **118**, 9481 (2003). Five TTG-graph variants are
built as separate executables; the integrals are computed on demand
via [libint2](https://github.com/evaleev/libint).

| Variant | Idea |
|---|---|
| v1 | bulk-synchronous TTG dispatch wrapped by a C++ outer loop |
| v2 | feedback edge folds the outer loop into a single TTG, single-column inner update |
| v3 | block-Cholesky-per-shell-pair, BLAS-3 gather |
| v4 | row-parallel gather: `gather_pivot` fans out N row-chunks as TTG tasks |
| v5 | data on the flow: per-chunk `L` is a bag of `Eigen::Tensor` tiles flowing on the loopback edge, distributable across MPI ranks |
| v6 | v5 + overlap-norm shell-pair screening (Almlöf-significant pairs only); brings the layout cost down from O(N²) to O(N) for extended molecules |

See `v[1-6].md` for the per-variant description. Shared building
blocks (basis / AO-pair / integral helpers, CLI parser, diagonal
compute, dense-`M` validation against `libint2::pivoted_cholesky`)
live in `common.{h,cc}`.

## Build

The example targets are added when `libint2` (≥ 2.10.0) is found by
CMake. v3 and the optional `eri-pch_v4_blas` target additionally route
Eigen's level-3 ops through a system BLAS (Apple Accelerate on macOS,
OpenBLAS / MKL on Linux):

```sh
ninja eri-pch_v1-parsec eri-pch_v2-parsec eri-pch_v3-parsec \
      eri-pch_v4-parsec eri-pch_v4_blas-parsec \
      eri-pch_v5-parsec eri-pch_v6-parsec
```

## Run

```sh
./examples/eri-pch_v<n>-parsec <xyz-file> <basis-name>
```

Each variant emits `eri-pch_v<n>.dot` (raw TTG flow graph) on
completion and validates the recovered `L Lᵀ` against
`libint2::pivoted_cholesky` on the dense `M`.

v6 reads the env var `TTG_ERI_PCH_SP_THRESHOLD` to override the
default shell-pair screening threshold (`1e-12`); set it to `0` to
reproduce v5's unscreened layout.
