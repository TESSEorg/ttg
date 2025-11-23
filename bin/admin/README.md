<!-- SPDX-License-Identifier: BSD-3-Clause -->

# TTG Admin Scripts

This directory contains administrative scripts for maintaining the TTG project.

## License Header Maintenance

### update-license-headers.sh

This script adds SPDX license identifiers to all source files in the repository.

**Usage:**

```bash
# Check which files are missing SPDX headers (non-destructive)
./bin/admin/update-license-headers.sh --check-only

# Add SPDX headers to all files missing them
./bin/admin/update-license-headers.sh
```

**What it does:**
- Scans all source files tracked by git
- Automatically excludes:
  - External code (ttg/ttg/external/)
  - Binary files (SSH keys, etc.)
  - Build directories
- Adds `SPDX-License-Identifier: BSD-3-Clause` using appropriate comment syntax:
  - C/C++/CUDA files: `// SPDX-License-Identifier: BSD-3-Clause`
  - CMake/Shell/Python files: `# SPDX-License-Identifier: BSD-3-Clause`
- Preserves shebang lines in scripts
- Skips files that already have SPDX headers

**When to use:**
- Before releases to ensure all new files have proper license headers
- In CI/CD to verify license compliance (use `--check-only`)
- After adding new source files to the repository

**CI Integration:**

Add this to your CI workflow to ensure all files have license headers:

```yaml
- name: Check license headers
  run: ./bin/admin/update-license-headers.sh --check-only
```

## Other Scripts

### make_boost_snapshot.sh

Creates a snapshot of Boost.CallableTraits for bundling with TTG.
This is used to maintain the vendored copy in `ttg/ttg/external/boost/`.
