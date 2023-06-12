# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

JUPYTEXT      ?= jupytext
NBCONVERT     ?= jupyter nbconvert
DEMOFOLDER   = demos/demos
OUTFOLDER    = demos/out

ADD_DOWNLOAD_BUTTONS ="$\
\$$a$\
:download:\`Download Python source code \<$$(basename $$1 .rst).py\>\`\n\n$\
:download:\`Download as Jupyter Notebook \<$$(basename $$1 .rst).ipynb\>\`\n"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 0. create the output folder and copy all Python source files over to it
# 1. find all Python demos and convert them to Jupyter notebooks
# 2. find all Jupyter notebooks and convert them into Sphinx rst files
# 3. find all rst files and add the file name as a label at the top
# 4. find all rst files and add the download links to the bottom
.PHONY: demos
demos:
	mkdir -p $(OUTFOLDER)
	cp -a $(DEMOFOLDER)/. $(OUTFOLDER)

	find $(DEMOFOLDER) -name '*.py' -exec sh -c '$(JUPYTEXT) --to ipynb $$1 --output "$(OUTFOLDER)/$$(basename $$1 .py).ipynb"' sh {} \;
	find $(OUTFOLDER) -name '*.ipynb' -exec $(NBCONVERT) {} --to rst --output-dir=$(OUTFOLDER) --execute --RegexRemovePreprocessor.patterns="^%" \;
	find $(OUTFOLDER) -name '*.rst' -exec sh -c 'sed -i "1s/^/.. _$$(basename $$1 .rst):\n\n/" $$1' sh {} \;

	find $(OUTFOLDER) -name '*.rst' -exec sh -c 'sed -i -e $(ADD_DOWNLOAD_BUTTONS) $$1' sh {} \;

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
