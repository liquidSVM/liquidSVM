default: lib

MAIN_DIR=../..
COMMON_DIR=$(MAIN_DIR)/bindings/common

SRC_DIR=liquidSVM
PKG_NAME=liquidSVM
PKG_NAME_VERSION=$(PKG_NAME)-$(VERSION)
PKG_NAME_GENERAL=$(PKG_NAME)-python
BUILD_DIR=build/$(SRC_DIR)

## needs also to be set in setup.py
VERSION=1.0.1

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

binary: prepare-src
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
	PYTHONPATH=$(PWD) $(python) -M coverage run test/testLiquidSVM.py
	$(python) -M coverage html

doc: doc/_build/html/index.html
doc/_build/html/index.html: liquidSVM/*.py doc/*.rst
	$(SPHINXBUILD) -M html "doc" "doc/_build" $(SPHINXOPTS)

pydoc: pydoc.html
pydoc.html: liquidSVM/__init__.py
	$(pydoc) -w liquidSVM/__init__.py
	mv __init__.html pydoc.html

SSH_MAC=isamac
SSH_MAC_DIR=work/python
SSH_WIN=isawinwin
SSH_WIN_DIR=isawin/work/python

BDIST_MAC=$(PKG_NAME_GENERAL).macosx.tar.gz
BDIST_WIN=$(PKG_NAME_GENERAL).win-amd64.zip

binaries: pack binaries-mac binaries-win

binaries-mac: #pack
		ssh -x $(SSH_MAC) 'mkdir -p $(SSH_MAC_DIR)'
		scp dist/$(PKG_NAME_VERSION).tar.gz $(SSH_MAC):$(SSH_MAC_DIR)/
		ssh -x $(SSH_MAC) 'cd $(SSH_MAC_DIR); rm -r $(PKG_NAME_VERSION); tar xzf $(PKG_NAME_VERSION).tar.gz; cd $(PKG_NAME_VERSION); LIQUIDSVM_CONFIGURE_ARGS=generic python setup.py bdist'
		scp $(SSH_MAC):$(SSH_MAC_DIR)/$(PKG_NAME_VERSION)/dist/$(PKG_NAME_VERSION)*macosx*.tar.gz dist/$(BDIST_MAC)

binaries-win:
		ssh -x $(SSH_WIN) 'mkdir -p $(SSH_WIN_DIR)'
		scp dist/$(PKG_NAME_VERSION).tar.gz $(SSH_WIN):$(SSH_WIN_DIR)/
		ssh -x $(SSH_WIN) 'cd $(SSH_WIN_DIR); rm -r $(PKG_NAME_VERSION); tar xzf $(PKG_NAME_VERSION).tar.gz; cd $(PKG_NAME_VERSION); LIQUIDSVM_CONFIGURE_ARGS=generic python setup.py bdist'
		scp $(SSH_WIN):$(SSH_WIN_DIR)/$(PKG_NAME_VERSION)/dist/$(PKG_NAME_VERSION).win-amd64.zip dist/$(BDIST_WIN)

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
