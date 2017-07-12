default: lib

MAIN_DIR=../..
COMMON_DIR=$(MAIN_DIR)/bindings/common

SRC_DIR=liquidSVM
PKG_NAME=liquidSVM
PKG_NAME_VERSION=$(PKG_NAME)-$(VERSION)
PKG_NAME_GENERAL=$(PKG_NAME)-python
BUILD_DIR=build/$(SRC_DIR)

## needs also to be set in setup.py
VERSION=$(shell sed -n 's/__version__ = "\(.*\)"/\1/p' setup.py)

PUB_DIR=$(MAIN_DIR)/pub/python

python=python
pip=pip
jupyter=jupyter
pydoc=pydoc
twine=twine

SPHINXBUILD   = sphinx-build
SPHINXOPTS    =


rm=\rm
# Makefiles echo somehow does not know -e ???
echo=/bin/echo

LIB_NAME=libliquidsvm.so
MODULE_NAME=liquidSVM

lib: bin/$(LIB_NAME)
	cp $(MODULE_NAME).py bin/

bin/$(LIB_NAME): 
	mkdir -p bin
	g++ -shared -fPIC -march=native -O3 -g -lrt -I$(MAIN_DIR) -o bin/$(LIB_NAME) $(COMMON_DIR)/liquidSVM.cpp -w -DCOMPILE_FOR_COMMAND_LINE__

prepare-src:
	\rm -r src/common src/sources 2>/dev/null || :
	mkdir -p src/common
	cp $(COMMON_DIR)/scenario_config.cpp $(COMMON_DIR)/scenario_config.h src/common
	cp $(COMMON_DIR)/liquidSVM.cpp $(COMMON_DIR)/liquidSVM.h src/common
	cp -r ../../sources src
	rm -r src/sources/*/object_code
pack: prepare-src README.txt
	$(python) setup.py sdist
	cp dist/$(PKG_NAME_VERSION).tar.gz dist/$(PKG_NAME_GENERAL).tar.gz
sdist: pack

bdist: prepare-src
	$(python) setup.py bdist

install:
	$(python) setup.py install --user

demo: install #lib
	mkdir -p bin
	cd bin && $(python) -m $(MODULE_NAME)

README.html index.html README.txt README.rst: README.md $(COMMON_DIR)/doc/global-and-grid.md
	cat README.md $(COMMON_DIR)/doc/global-and-grid.md \
	| pandoc -t html --standalone \
	> README.html
	cat README.md | pandoc --template $(COMMON_DIR)/doc/frag.template | sed 's/\[/\&#x5b;/g' > index.html
	@#cat README.md | pandoc -t plain > README.txt
	cat README.md | pandoc -t rst > README.rst

demo.html: demo.ipynb
	$(jupyter) nbconvert --to html --template full demo.ipynb

unittest:
	PYTHONPATH=$(PWD) $(python) test/testLiquidSVM.py

coverage:
	PYTHONPATH=$(PWD) $(python) -m coverage run test/testLiquidSVM.py
	$(python) -m coverage html

doc: doc/_build/html/index.html
doc/_build/html/index.html: liquidSVM/*.py doc/*.rst
	$(SPHINXBUILD) -M html "doc" "doc/_build" $(SPHINXOPTS)

pydoc: pydoc.html
pydoc.html: liquidSVM/__init__.py
	$(pydoc) -w liquidSVM/__init__.py
	mv __init__.html pydoc.html

SSH_MAC=mus01
SSH_MAC_DIR=work/python
# condapy3.3 gives a problem on both mac and windows at the moment.
# should we also do manylinux1: https://github.com/pypa/manylinux ???
ANACONDA_BUILD_ENVS=condapy2 condapy3 condapy3.4 condapy3.5
SSH_MAC_CONDA_HOME=~/anaconda
#SSH_MAC_PIP2=~/anaconda/bin/pip
#SSH_MAC_PIP3=~/anaconda/bin/pip
#SSH_MAC_PIP2=venv/py2/bin/pip
#SSH_MAC_PIP3=venv/py3/bin/pip

SSH_WIN=isawinwin
SSH_WIN_DIR=winhome2/work/python
SSH_WIN_CONDA_HOME=~/winhome2/Miniconda3
#SSH_WIN_PIP2=venv/py2/Scripts/pip
#SSH_WIN_PIP3=venv/py3/Scripts/pip

BDIST_MAC=$(PKG_NAME_GENERAL).macosx.tar.gz
BDIST_WIN=$(PKG_NAME_GENERAL).win-amd64.zip

binaries: pack binaries-mac binaries-win

prepare-win: #pack
	ssh -x $(SSH_WIN) 'mkdir -p $(SSH_WIN_DIR)'
	scp dist/$(PKG_NAME_VERSION).tar.gz $(SSH_WIN):$(SSH_WIN_DIR)/

prepare-mac: #pack
	ssh -x $(SSH_MAC) 'mkdir -p $(SSH_MAC_DIR)'
	scp dist/$(PKG_NAME_VERSION).tar.gz $(SSH_MAC):$(SSH_MAC_DIR)/

binaries-mac: prepare-mac
	ssh -x $(SSH_MAC) 'cd $(SSH_MAC_DIR); export LIQUIDSVM_CONFIGURE_ARGS=generic ; ' \
	'for i in $(ANACONDA_BUILD_ENVS); do echo +++ making python wheel for $$i; $(SSH_MAC_CONDA_HOME)/envs/$$i/bin/pip wheel --no-deps $(PKG_NAME_VERSION).tar.gz; done'
	scp $(SSH_MAC):$(SSH_MAC_DIR)/$(PKG_NAME_VERSION)*.whl dist/

binaries-win: prepare-win
	ssh -x $(SSH_WIN) 'cd $(SSH_WIN_DIR); export LIQUIDSVM_CONFIGURE_ARGS=generic ; ' \
	'for i in $(ANACONDA_BUILD_ENVS); do echo +++ making python wheel for $$i; $(SSH_WIN_CONDA_HOME)/envs/$$i/Scripts/pip wheel --no-deps $(PKG_NAME_VERSION).tar.gz; done'
	scp $(SSH_WIN):$(SSH_WIN_DIR)/$(PKG_NAME_VERSION)-*-win_amd64.whl dist/

pub: dist/$(PKG_NAME).tar.gz pub-quick

pub-quick: index.html #pack
	mkdir -p $(PUB_DIR)
	cp index.html dist/$(PKG_NAME_VERSION).tar.gz dist/$(PKG_NAME_GENERAL).tar.gz $(PUB_DIR)
	if [[ -r dist/$(BDIST_WIN) ]]; then cp dist/$(BDIST_WIN) $(PUB_DIR); fi
	if [[ -r dist/$(BDIST_MAC) ]]; then cp dist/$(BDIST_MAC) $(PUB_DIR); fi

twine:
	@#$(twine) register dist/$(PKG_NAME_VERSION).tar.gz
	$(twine) upload dist/$(PKG_NAME_VERSION).tar.gz
	echo also for mac/win wheels: $(twine) upload dist/$(PKG_NAME_VERSION).tar.gz

regenerate-quick:
	$(echo) -e "### This file is generated do not edit it by hand!\n\n" > liquidSVM/doc.py
	$(echo) "def configuration():" >> liquidSVM/doc.py
	$(echo) -n '    """' >> liquidSVM/doc.py
	pandoc -t rst  $(COMMON_DIR)/doc/global-and-grid.md | sed -e "s/^/    /" \
		| sed -e 's/c(\([^)]*\))/[\1]/' >>  liquidSVM/doc.py
	$(echo) -e '    """\n\n' >> liquidSVM/doc.py
	for i in train select test; do \
		$(echo) -e "def $${i}Args():\n    '''Arguments for SVM.$$i" >> liquidSVM/doc.py; \
		pandoc -t rst  $(COMMON_DIR)/doc/md/command-args-svm-$$i.md | sed -e "s/^/    /" \
		| sed -e 's/c(\([^)]*\))/[\1]/' >>  liquidSVM/doc.py ; \
		$(echo) -e "    '''\n\n" >> liquidSVM/doc.py ; \
	done


clean:
	$(rm) -rf bin/* src/sources src/common
