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

# Script to add download buttons to the bottom of the rst files
ADD_DOWNLOAD_BUTTONS ="$\
\$$a$\
:download:\`Download Python source code \<$$(basename $$1 .rst).py\>\`\n\n$\
:download:\`Download as Jupyter Notebook \<$$(basename $$1 .rst).ipynb\>\`\n"

# Script to add file names as labels to the top of the rst files
ADD_FILE_NAME_LABEL = "1s/^/.. _$$(basename $$1 .rst):\n\n/"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: demos
demos:
	# 0. create the output folder and copy all Python source files over to it
	mkdir -p $(OUTFOLDER)
	cp -a $(DEMOFOLDER)/. $(OUTFOLDER)

	# 1. find all Python demos and convert them to Jupyter notebooks
	find $(DEMOFOLDER) -name '*.py' -exec sh -c '$(JUPYTEXT) --to ipynb $$1 --output "$(OUTFOLDER)/$$(basename $$1 .py).ipynb"' sh {} \;

	# 2. find all Jupyter notebooks and convert them into Sphinx rst files
	find $(OUTFOLDER) -name '*.ipynb' -exec $(NBCONVERT) {} --to rst --output-dir=$(OUTFOLDER) --execute --RegexRemovePreprocessor.patterns="^%" \;

	# 3. find all rst files and, add file name labels, replace major classes/methods with Sphinx hyperlinks, and add download links
	find $(OUTFOLDER) -name '*.rst' -exec sh -c 'sed --in-place \
		-e $(ADD_FILE_NAME_LABEL) \
		-e "s/\`\`Circuit\`\`/:class:\`~dwave.gate.circuit.Circuit\`/g" \
		-e "s/\`\`Circuit\.\(\w*\)\`\`/:meth:\`~dwave.gate.circuit.Circuit.\1\`/g" \
		-e "s/\`\`operations\`\`/:mod:\`~dwave.gate.operations\`/g" \
		-e "s/\`\`ops\.\(\w*\)\`\`/:class:\`~dwave.gate.operations.operations.\1\`/g" \
		-e $(ADD_DOWNLOAD_BUTTONS) \
	$$1' sh {} \; \

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
