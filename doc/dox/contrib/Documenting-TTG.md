# Documenting TTG {#Documenting-TTG}

## Documenting Source

TTG C/C++ source should be documented using [Doxygen](https://www.doxygen.nl).
Doxygen supports in-source documentation and stand-alone documents.
TTG's Doxygen configuration is contained in the [Doxyfile.in](https://github.com/TESSEorg/ttg/blob/master/doc/dox/config/Doxyfile.in).
TTG enables support for Markdown both in in-source comments
and for Markdown pages (with `.md` extension). The use of Markdown is encouraged.

## Administration

- Generation and deployment of TTG's documentation is performed by successful
CI jobs defined [here](https://github.com/TESSEorg/ttg/blob/master/.github/workflows/cmake.yml). The `Build+Deploy Dox`
step assembles [the TTG website](https://tesseorg.github.io/ttg) by combining
  - the frontmatter: currently just TTG's [README.md](https://github.com/TESSEorg/ttg/blob/master/README.md) file),
  - the rest of the website content located on the [gh-pages-template branch](https://github.com/TESSEorg/ttg/tree/gh-pages-template)
of the TTG repo
  - the Doxygen html documentation
- Dox deployment uses a GitHub token that is defined as variable `GH_TTG_TOKEN` in GHA's TTG repo settings' [secrets](https://github.com/TESSEorg/ttg/settings/secrets/actions).
