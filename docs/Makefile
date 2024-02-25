# Minimal makefile for Sphinx documentation
# This makefile is used to build Sphinx documentation with the help of the
# sphinx-build script. It is a minimal configuration and can be customized
# according to specific needs.

# You can set these variables from the command line.
# These variables can be overwritten from the command line when calling the
# make command.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build   # The sphinx-build script, used to build the docs
SOURCEDIR     = source         # The source directory containing the reStructuredText files
BUILDDIR      = _build         # The build directory where the built documentation will be placed

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# The 'help' target shows the list of available targets and their descriptions

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# This rule ensures that any unknown target will be treated as a command to
# build the documentation using the sphinx-build script with the make mode.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
