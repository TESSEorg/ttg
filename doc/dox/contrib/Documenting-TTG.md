# Documenting TTG {#Documenting-TTG}

## Documenting Source

TTG C/C++ source should be documented using [Doxygen](www.doxygen.nl).
Doxygen supports in-source documentation and stand-alone documents.
TTG's Doxygen configuration is contained in the [Doxyfile.in](github.com/TESSEorg/ttg/blob/master/doc/dox/config/Doxyfile.in).
TTG enables support for Markdown both in in-source comments
and for Markdown pages (with `.md` extension). The use of Markdown is encouraged.

## Administration

- Generation and deployment of TTG's documentation is performed by successful
Travis-CI jobs, using [this script](github.com/TESSEorg/ttg/blob/master/bin/deploy-linux.sh). The script
assembles [the TTG website](tesseorg.github.io/ttg) by combining
  - the frontmatter: currently just TTG's [README.md](https://github.com/TESSEorg/ttg/blob/master/README.md) file),
  - the rest of the website content located on the [gh-pages-template branch](github.com/TESSEorg/ttg/tree/gh-pages-template)
of the TTG repo
  - the Doxygen html documentation
- The deployment script is invoked via [.travis.yml](github.com/TESSEorg/ttg/blob/master/.travis.yml).
