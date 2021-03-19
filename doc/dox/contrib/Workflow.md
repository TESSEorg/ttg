# Recommended Workflow Elements {#Recommended-Workflow-Elements}

## `pre-commit` git hooks

It is recommended to use the [pre-commit hook manager](https://pre-commit.com/) to enforce coding conventions, perform
static code analysis, and manage TTG-specific infrastructure. Simply install `pre-commit` as
described [here](https://pre-commit.com/#installation). Then run `pre-commit install` in the TTG source directory.
File `.pre-commit-config.yaml` describes the hook configuration used by TTG; feel free to PR additional hooks.

Each time you try to commit a changeset in a repo in which `pre-commit` hooks have been installed each hook will be
executed on each file added or changed in the changeset. Some hooks are designed to simply prevent nonconformant source
code, documentation, infrastructure files, etc. from being committed, whereas other hooks will change the files to make
them conformant. In either case, the commit will fail if any changes are needed. You will need to update the changeset (
by amending the commit with the changes performed by the hooks and/or any changes you performed manually) and try again.

N.B. Changes in files performed by the `pre-commit` hooks are not instantly "seen" by the IDE, so it is recommended to
manually run `git status` after a failed commit.

The most important use case for `pre-commit` hooks is invoking clang-format automatically on each file added or changed
in a commit

### pre-commit git hook: clang-format

This hook runs `clang-format` to enforce the TTG code formatting conventions.
