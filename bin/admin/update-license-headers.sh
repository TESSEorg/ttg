#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
#
# Script to add or update SPDX license headers in TTG source files
# Usage: ./update-license-headers.sh [--check-only]
#
# This script:
# - Finds all source files tracked by git
# - Excludes external code (ttg/ttg/external/)
# - Excludes binary files and non-code files
# - Adds SPDX-License-Identifier: BSD-3-Clause where missing
# - Uses appropriate comment syntax for each file type

set -e

# Check if running from repository root
if [ ! -d ".git" ]; then
  echo "Error: This script must be run from the repository root"
  exit 1
fi

CHECK_ONLY=0
if [ "$1" = "--check-only" ]; then
  CHECK_ONLY=1
  echo "Running in check-only mode..."
fi

# Create temporary Python script to do the work
SCRIPT=$(mktemp -t add_spdx_headers.XXXXXX).py
cat > "$SCRIPT" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Add SPDX license headers to source files."""

import os
import sys
import re

CHECK_ONLY = len(sys.argv) > 1 and sys.argv[1] == "--check-only"

def get_spdx_header(filepath):
    """Return the appropriate SPDX header for the given file."""
    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath)

    # Shebang files - check the first line
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#!'):
                # Shell or Python script with shebang
                return '# SPDX-License-Identifier: BSD-3-Clause\n'
    except:
        pass

    # CMake files use # for comments
    if ext == '.cmake' or basename == 'CMakeLists.txt':
        return '# SPDX-License-Identifier: BSD-3-Clause\n'

    # C/C++ files use // for comments
    if ext in ['.h', '.hpp', '.cpp', '.cc', '.cxx', '.c', '.cu', '.hip']:
        return '// SPDX-License-Identifier: BSD-3-Clause\n'

    # .in files - need to check the base filename to determine type
    if ext == '.in':
        # Check if it's a C/C++ header template (*.h.in, *.hpp.in, etc.)
        base_name = os.path.basename(filepath)
        if base_name.endswith('.h.in') or base_name.endswith('.hpp.in'):
            return '// SPDX-License-Identifier: BSD-3-Clause\n'
        # Otherwise assume it's a CMake or config file template
        return '# SPDX-License-Identifier: BSD-3-Clause\n'

    # Python files use # for comments
    if ext in ['.py']:
        return '# SPDX-License-Identifier: BSD-3-Clause\n'

    # Shell scripts use # for comments
    if ext in ['.sh', '.bash']:
        return '# SPDX-License-Identifier: BSD-3-Clause\n'

    # Default to C++ style for unknown types
    return '// SPDX-License-Identifier: BSD-3-Clause\n'

def has_spdx_header(content):
    """Check if the file already has an SPDX header."""
    lines = content.split('\n', 10)  # Check first 10 lines
    for line in lines:
        if 'SPDX-License-Identifier' in line:
            return True
    return False

def add_spdx_header(filepath):
    """Add SPDX header to the file if it doesn't already have one."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False, False

    # Check if already has SPDX header
    if has_spdx_header(content):
        return True, False  # success, but no change needed

    # Get the appropriate header
    header = get_spdx_header(filepath)

    # For files with shebang, add after shebang
    lines = content.split('\n', 1)
    if lines[0].startswith('#!'):
        if len(lines) > 1:
            new_content = lines[0] + '\n' + header + '\n' + lines[1]
        else:
            new_content = lines[0] + '\n' + header
    else:
        new_content = header + content

    if CHECK_ONLY:
        print(f"Missing SPDX header: {filepath}")
        return False, True

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Added SPDX header to {filepath}")
        return True, True
    except Exception as e:
        print(f"Error writing {filepath}: {e}", file=sys.stderr)
        return False, False

# Read file list from stdin
files = [line.strip() for line in sys.stdin if line.strip()]

success_count = 0
skip_count = 0
fail_count = 0
missing_count = 0

for filepath in files:
    success, changed = add_spdx_header(filepath)
    if success and changed:
        success_count += 1
    elif success and not changed:
        skip_count += 1
    elif not success and changed:
        missing_count += 1
    else:
        fail_count += 1

if CHECK_ONLY:
    print(f"\nCheck complete:")
    print(f"  Files with SPDX headers: {skip_count}")
    print(f"  Files missing SPDX headers: {missing_count}")
    if missing_count > 0:
        sys.exit(1)
else:
    print(f"\nProcessed {len(files)} files:")
    print(f"  Added headers: {success_count}")
    print(f"  Already had headers: {skip_count}")
    print(f"  Failed: {fail_count}")
PYTHON_SCRIPT

chmod +x "$SCRIPT"

# Find all source files, excluding:
# - External code (ttg/ttg/external/)
# - Build directories
# - Git directory
# - Binary files (SSH keys, etc.)
# - IDE files
git ls-files \
  '*.h' '*.hpp' '*.cpp' '*.cc' '*.cxx' '*.c' '*.cu' '*.hip' \
  '*.cmake' '*.in' 'CMakeLists.txt' \
  '*.sh' '*.bash' '*.py' \
  | grep -v '^ttg/ttg/external/' \
  | grep -v '\.id_rsa' \
  | grep -v '/build/' \
  | python3 "$SCRIPT" $([ $CHECK_ONLY -eq 1 ] && echo "--check-only" || echo "")

EXIT_CODE=$?

# Cleanup
rm -f "$SCRIPT"

if [ $CHECK_ONLY -eq 1 ]; then
  if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ All source files have SPDX license headers"
  else
    echo ""
    echo "✗ Some files are missing SPDX license headers"
    echo "  Run without --check-only to add them automatically"
  fi
else
  echo ""
  echo "Done! Review changes with: git diff"
fi

exit $EXIT_CODE
