<!-- SPDX-License-Identifier: BSD-3-Clause -->

# Contributing to TTG

Thank you for your interest in contributing to TTG (Template Task Graph)!

TTG is an academic research project developed collaboratively by researchers from Stony Brook University, University of Tennessee Knoxville, and Virginia Tech. We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Contributing Code](#contributing-code)
- [Development Workflow](#development-workflow)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Building and Testing](#building-and-testing)
  - [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). All contributors are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/TESSEorg/ttg/issues) to avoid duplicates.

When filing a bug report, please include:

- **Clear descriptive title**
- **Detailed description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs. actual behavior
- **Environment information**:
  - TTG version (commit hash or release)
  - Operating system and version
  - Compiler and version
  - CMake version
  - Backend (PaRSEC/MADNESS) and version
- **Minimal code example** that reproduces the issue (if applicable)
- **Error messages** or output (use code blocks for formatting)

Use the bug report template when creating an issue.

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- Use the feature request template
- Provide a clear use case for the enhancement
- Explain why this enhancement would be useful to most TTG users
- Consider whether it fits within the scope and design philosophy of TTG

### Contributing Code

We welcome code contributions! Common areas for contribution include:

- Bug fixes
- Performance improvements
- Documentation improvements
- Example programs
- Test coverage
- Support for new platforms or compilers
- New features (please discuss via an issue first)

## Development Workflow

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ttg.git
   cd ttg
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/TESSEorg/ttg.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Building and Testing

Detailed build instructions are available in [INSTALL.md](INSTALL.md).

**Quick start:**

```bash
# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DTTG_EXAMPLES=ON

# Build
cmake --build build

# Run tests
cd build
ctest -V
```

**Before submitting a pull request:**

- Ensure all existing tests pass
- Add tests for new features or bug fixes
- Test with both PaRSEC and MADNESS backends (if applicable)
- Test on multiple compilers if possible (GCC, Clang)

### Code Style Guidelines

#### General Principles

- Write clear, readable, and maintainable code
- Follow existing code style in the file you're editing
- Keep functions focused and reasonably sized
- Use meaningful variable and function names

#### C++ Style

- **Standard**: C++20
- **Naming conventions**:
  - Types (classes, structs): `PascalCase` or `snake_case` (follow existing patterns)
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE` or `kCamelCase`
  - Template parameters: `CamelCase` with trailing `T` suffix common
- **Formatting**:
  - Indentation: 2 spaces (no tabs)
  - Line length: aim for 120 characters max, but prioritize readability
  - Braces: follow existing style in the file
- **Comments**:
  - Use `//` for single-line comments
  - Use Doxygen-style comments for public APIs
  - Explain *why*, not just *what*
- **Headers**:
  - All source files must have SPDX license identifier:
    ```cpp
    // SPDX-License-Identifier: BSD-3-Clause
    ```
  - Use header guards (`#ifndef`/`#define`/`#endif`)
  - Include what you use

#### CMake Style

- Use lowercase for function names: `add_library()`, `target_link_libraries()`
- Use `${VAR}` for variable references
- Indent with 2 spaces
- All CMake files must have SPDX license identifier:
  ```cmake
  # SPDX-License-Identifier: BSD-3-Clause
  ```

#### Documentation

- Public APIs should have Doxygen comments
- Complex algorithms should have explanatory comments
- Update relevant documentation when changing behavior
- Add or update examples when adding features

## Pull Request Process

1. **Keep changes focused**: One pull request = one feature/fix
2. **Update documentation**: Ensure docs reflect your changes
3. **Add tests**: New features need tests
4. **Run tests locally**: Ensure all tests pass before submitting
5. **Add SPDX headers**: Use `bin/admin/update-license-headers.sh` to verify
6. **Write a clear PR description**:
   - What problem does this solve?
   - How does it solve it?
   - Any breaking changes?
   - Related issues (use "Fixes #123" to auto-close)
7. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
8. **Respond to review feedback**: Be open to suggestions and changes
9. **Squash commits**: Consider squashing related commits before merging

### PR Review Process

- A project maintainer will review your PR
- Reviews may take time - this is an academic project with volunteers
- Address feedback and update your PR
- Once approved, a maintainer will merge your PR

### Continuous Integration

Pull requests automatically run:
- Build tests on multiple platforms
- Unit tests
- Code formatting checks (planned)
- License header checks (planned)

All checks must pass before merging.

## Getting Help

- **Issues**: Use [GitHub Issues](https://github.com/TESSEorg/ttg/issues) for bug reports and feature requests
- **Discussions**: Use [GitHub Discussions](https://github.com/TESSEorg/ttg/discussions) for questions and general discussion
- **Documentation**: See the [online documentation](https://tesseorg.github.io/ttg/)

## Recognition

All contributors are recognized in the project's git history and on the [GitHub contributors page](https://github.com/TESSEorg/ttg/graphs/contributors). Significant contributors may be acknowledged in the [COPYRIGHT](COPYRIGHT) file.

## License

By contributing to TTG, you agree that your contributions will be licensed under the [BSD 3-Clause License](LICENSE).

---

Thank you for contributing to TTG! 🎉
