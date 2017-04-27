default: pack

MAIN_DIR=../..
SRC_DIR=.
COMMON_BASE_DIR=$(MAIN_DIR)/bindings
COMMON_DIR=$(COMMON_BASE_DIR)/common

MATLAB=matlab -nodesktop

PKG_NAME=liquidSVM.mltbx
SRC_PKG_NAME=liquidSVM-octave.zip
PROJECT_FILE=liquidSVM.prj
BUILD_DIR=.

PUB_DIR=$(MAIN_DIR)/pub/matlab

rm=\rm


README.html index.html: README.md $(COMMON_DIR)/doc/global-and-grid.md
	cat README.md $(COMMON_DIR)/doc/global-and-grid.md \
	| pandoc -t html --standalone \
	> README.html
	cat README.md | pandoc --template $(COMMON_DIR)/doc/frag.template | sed 's/\[/\&#x5b;/g' > index.html
	mkdir -p doc
	cp README.html doc/
	cat README.md | pandoc -t plain > README.txt


pack: $(PKG_NAME)

$(PKG_NAME): pack-prepare
	# check the paths .../liquidSVM/bindings/matlab !!
	$(MATLAB) -r "matlab.addons.toolbox.packageToolbox('$(PROJECT_FILE)'); exit;"

pack-src: pack-prepare
	## these all should be in a directory?
	zip -r $(SRC_PKG_NAME) classes functions *.m mexliquidSVM.cpp liquidSVM.prj data.mat README.txt README.html src COPYING Makefile

pack-prepare: README.txt README.html
	#cp -pr README.txt $(BUILD_DIR)
	cp -pr $(MAIN_DIR)/COPYING $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/src/common
	cp -pr $(COMMON_DIR)/liquidSVM.* $(COMMON_DIR)/scenario_config.* $(BUILD_DIR)/src/common/
	cp -pr $(MAIN_DIR)/sources $(BUILD_DIR)/src/
	rm -f $(BUILD_DIR)/src/sources/*/object_code/*

#unittest-octave:
	#$(MATLAB) -r "disp(runtests()); exit;"

unittest:
	$(MATLAB) -r "disp(runtests()); exit;"

SSH_MAC=isamac
SSH_MAC_DIR=work
SSH_WIN=isawinwin
SSH_WIN_DIR=isawin/work

binaries: pack-quick binaries-mac binaries-win javadoc pack-quick

binaries-mac: #pack
		scp build/$(PKG_NAME).zip $(SSH_MAC):$(SSH_MAC_DIR)/
		ssh -x $(SSH_MAC) 'cd $(SSH_MAC_DIR); rm -r $(PKG_NAME); unzip -q $(PKG_NAME).zip; cd $(PKG_NAME); make generic'
		scp $(SSH_MAC):$(SSH_MAC_DIR)/$(PKG_NAME)/$(LIBNAME_MAC) bin/$(LIBNAME_MAC)

binaries-win:
		scp build/$(PKG_NAME).zip $(SSH_WIN):$(SSH_WIN_DIR)/
		ssh -x $(SSH_WIN) 'cd $(SSH_WIN_DIR); rm -r $(PKG_NAME); unzip -q $(SRC_PKG_NAME); cd $(SRC_PKG_NAME); make generic'
		scp $(SSH_WIN):$(SSH_WIN_DIR)/$(PKG_NAME)/$(LIBNAME_WIN) bin/$(LIBNAME_WIN)

pub: $(PKG_NAME) $(SRC_PKG_NAME) pub-quick

pub-quick: index.html
	mkdir -p $(PUB_DIR)
	cp index.html $(PKG_NAME) $(SRC_PKG_NAME) $(PUB_DIR)

clean:
	$(rm) -rf src COPYING

#######  Octave specifics  #######

OCTAVE=octave-cli

octaveTest: octaveTestliquidSVM.m
	$(OCTAVE) --eval test\ octaveTestliquidSVM.m

octaveTestliquidSVM.m: testliquidSVM.m
	cat testliquidSVM.m | sed -e 's/^/%! /' | \
	sed 's/%! function test_/%!test %/' | \
	sed 's/%! function setupOnce(d)/%!shared d %/' | \
	grep -v '%! function' | grep -v '%! end' | \
	cat > octaveTestliquidSVM.m
