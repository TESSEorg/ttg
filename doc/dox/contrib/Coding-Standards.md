# Coding Standards {#Coding-Standards}

## Code Organization

### Logical organization

TTG namespaces:

- `ttg`: contains top-level runtime-agnostic components as well as the default runtime-specific components
- `ttg_RUNTIME`: contains runtime-specific TTG components; two implementations are provided with TTG: `ttg_madness`
  and `ttg_parsec`

### Physical organization

Directory structure:

- `ttg`:
  - `ttg/ttg`: contains the entire TTG implementation
  - `ttg/ttg/base`: contains runtime-agnostic components
  - `ttg/ttg/RUNTIME`: contains TTG backend for specific runtime
  - `ttg/ttg/common`: contains runtime-specific components that are common to all runtimes
  - `ttg/ttg/util`: contains various utilities
- `tests`: contains unit tests
- `examples`: contains examples
