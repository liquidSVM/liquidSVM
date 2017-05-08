
default: regenerate update

PKG_NAME=liquidSVM

MAIN_DIR=../..
COMMON_DIR=$(MAIN_DIR)/bindings/common

MAC_HOST=isamac
MAC_DIR=work/R
MAC_R=R
MAC_R-devel=R
WIN_HOST=isawinwin
WIN_DIR=winhome2/work/R
WIN_R=R
WIN_R-devel=R
CUDA_TEST_HOST=isa11
CUDA_TEST_DIR=/usr/local/isa-scratch/work/R
CUDA_TEST_PATH=/usr/local/cuda
CUDA_TEST_Rscript=Rscript
CUDA_TEST_R=R


SRC_DIR=$(PKG_NAME)
BUILD_DIR=build/$(PKG_NAME)

VERSION=$(shell sed -n 's/Version: //p' $(SRC_DIR)/DESCRIPTION)

BUNDLE_SRC=$(PKG_NAME)_$(VERSION).tar.gz
BUNDLE_WIN=$(PKG_NAME)_$(VERSION).zip
BUNDLE_MAC=$(PKG_NAME)_$(VERSION).tgz

PUB_DIR=/afs/.mathe/public/www/isa/software/R
SCP_HOST=isa

PUB_SCP=$(SCP_HOST):$(PUB_DIR)

R=R
RCMD=$(R) CMD
rm=\rm
scp=scp
ssh=ssh


clean:
	$(rm) -fr build *.gz *.zip ${SRC_DIR}/src/*.{o,so,dynlib,dll}
	find $(SRC_DIR)/src -name \*.o -exec $(rm) '{}' \;

pack: pack-src
prepare-src: $(SRC_DIR)
	$(rm) -fr $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)
	cp -rp $(SRC_DIR)/* $(BUILD_DIR)
	$(rm) -fr $(BUILD_DIR)/src/sources $(BUILD_DIR)/$(PKG_NAME).Rproj $(BUILD_DIR)/src/*.{o,so,dynlib,dll} $(BUILD_DIR)/man-roxygen
	$(rm) -fr $(BUILD_DIR)/src/Makevars
	cp -a $(COMMON_DIR)/*.* $(BUILD_DIR)/src/
	cp -a $(MAIN_DIR)/sources $(BUILD_DIR)/src/
	$(rm) -fr $(BUILD_DIR)/src/sources/*/object_code/
	find $(BUILD_DIR) -name '.*' -exec $(rm) -r '{}' \;
	cp $(SRC_DIR)/.Rbuildignore $(BUILD_DIR)
	find $(BUILD_DIR) -name .directory -exec $(rm) '{}' \;
	find $(BUILD_DIR) -name \*.o -exec $(rm) '{}' \;
	find $(BUILD_DIR) -name \*.sol -exec $(rm) '{}' \;
	find $(BUILD_DIR) -name \*.fsol -exec $(rm) '{}' \;

pack-src: prepare-src
	$(rm) -fr build/$(BUNDLE_SRC)
	cd build && $(RCMD) build $(PKG_NAME)

pack-src-quick: prepare-src
	cd build && $(RCMD) build --no-build-vignettes $(PKG_NAME)

check: pack-src
	cd build && $(RCMD) check $(BUNDLE_SRC)

check-cran: pack-src
	cd build && $(RCMD) check --as-cran $(BUNDLE_SRC)

check-quick: pack-src-quick
	cd build && $(RCMD) check $(PKG_NAME)

pack-bin: prepare-src
	cd build && $(RCMD) INSTALL --build --preclean $(PKG_NAME)

demo: pack-src
	mkdir -p build/testlib
	cd build && $(RCMD) INSTALL -l testlib liquidSVM
	echo 'library(liquidSVM,lib.loc="build/testlib"); mean(iris$$Species==test( svm(Species~.,iris, d=1), iris ))' | $(R) --no-save

../../bin/svm-train:
	cd ../.. && make svm-train
../../bin/svm-select:
	cd ../.. && make svm-select
../../bin/svm-test:
	cd ../.. && make svm-test


$(SRC_DIR)/man-roxygen/template-parameters-%.R: $(COMMON_DIR)/doc/md/command-args-%.md
	echo "#'@"section Documentation for command-line parameters of `basename $@|sed 's/^.*\(svm-.*\)\.R$$/\1/'`: > $@
	echo "#'"The following parameters can be used as well: >> $@
	echo "#'\\itemize{" >> $@
	cat $< | \
	sed 's/^\* /\\item /' | \
	sed 's/`\([^`]*\)`/\\code{\1}/g' | \
	sed 's/^\s*//' | \
	sed 's/$$/\\cr/' | \
	sed 's/^.cr$$/\\ifelse{latex}{\\out{\\medskip}}{\\cr}/' | \
	sed "s/^/#'/" | \
	cat >> $@
	echo "#'}" >> $@

$(SRC_DIR)/man-roxygen/template-parameters-%.R.old: $(COMMON_DIR)/doc/md/command-args-%.md
	echo "#'@"section Documentation for command-line parameters of `basename $@|sed 's/^.*\(svm-.*\)\.R$$/\1/'`: > $@
	echo "#'"The following parameters can be used as well: >> $@
	@# | sed -e "s/^/#' /" |
	pandoc -t md2rd.lua $< | sed -e '1{/^$$/d}' | \
	sed '/^\\item/! s/$$/\\cr/' | \
	sed 's/^\\cr$$/\\ifelse{latex}{\\out{\\medskip}}{\\cr}/' | \
	sed "s/^/#'/" | \
	cat >> $@

template-parameters: $(SRC_DIR)/man-roxygen/template-parameters-svm-train.R $(SRC_DIR)/man-roxygen/template-parameters-svm-select.R $(SRC_DIR)/man-roxygen/template-parameters-svm-test.R

update: check update-quick

update-quick: pub binaries

pub:
	scp build/$(BUNDLE_SRC) $(SRC_DIR)/inst/doc/*.html $(PUB_SCP)

binaries: binaries-win binaries-mac

binaries-win:
	@echo === Make sure that directory $(WIN_DIR) on $(WIN_HOST) exists
	$(scp) build/$(BUNDLE_SRC) $(WIN_HOST):$(WIN_DIR)/
	$(ssh) $(WIN_HOST) cd $(WIN_DIR) \; \
		export LIQUIDSVM_CONFIGURE_ARGS=generic \; 'PATH=`echo $$PATH | sed "s/:/\n/g" | grep /cygdrive|paste -s -d:`' \
		$(WIN_R) CMD INSTALL --build $(BUNDLE_SRC)
	mkdir -p build/win/
	$(scp) $(WIN_HOST):$(WIN_DIR)/$(BUNDLE_WIN) build/win/

check-win:
	@echo === Make sure that directory $(WIN_DIR) on $(WIN_HOST) exists
	$(scp) build/$(BUNDLE_SRC) $(WIN_HOST):$(WIN_DIR)/
	$(ssh) $(WIN_HOST) cd $(WIN_DIR) \; 'PATH=`echo $$PATH | sed "s/:/\n/g" | grep /cygdrive|paste -s -d:`' $(WIN_R) CMD check $(BUNDLE_SRC)

binaries-mac:
	@echo === Make sure that directory $(MAC_DIR) on $(MAC_HOST) exists
	$(scp) build/$(BUNDLE_SRC) $(MAC_HOST):$(MAC_DIR)/
	$(ssh) $(MAC_HOST) cd $(MAC_DIR) \; $(MAC_R) CMD INSTALL --build --configure-args=generic $(BUNDLE_SRC)
	mkdir -p build/mac/
	$(scp) $(MAC_HOST):$(MAC_DIR)/$(BUNDLE_MAC) build/mac/

check-mac:
	@echo === Make sure that directory $(MAC_HOST) on $(MAC_HOST) exists
	$(scp) build/$(BUNDLE_SRC) $(MAC_HOST):$(MAC_DIR)/
	$(ssh) $(MAC_HOST) cd $(MAC_DIR) \; $(MAC_R) CMD check $(BUNDLE_SRC)

check-cuda:
	@echo === Make sure that directory $(CUDA_TEST_DIR) on $(CUDA_TEST_HOST) exists
	$(scp) build/$(BUNDLE_SRC) $(CUDA_TEST_HOST):$(CUDA_TEST_DIR)/
	$(ssh) -x $(CUDA_TEST_HOST) mkdir -p $(CUDA_TEST_DIR)/testlib/ \; cd $(CUDA_TEST_DIR) \; \
		$(CUDA_TEST_R) CMD INSTALL -l testlib --configure-args=$(CUDA_TEST_PATH) $(BUNDLE_SRC) \; \
		$(CUDA_TEST_Rscript) -e "library\(liquidSVM,lib=\'testlib\'\)" -e "lsSVM\(Y~.,\'covtype.1000\', d=1,GPUs=1\)"


check-all: check check-mac check-win #check-cuda

unittest:
	cd $(SRC_DIR) && Rscript -e 'devtools::test()'

coverage:
	cd $(SRC_DIR) && echo 'Sys.setenv(NOT_CRAN='true'); covr::package_coverage()' | $(R) --no-save

regenerate: template-parameters regenerate-quick

regenerate-quick:
	cp $(COMMON_DIR)/doc/*.md $(SRC_DIR)/vignettes/md/
	mkdir -p $(SRC_DIR)/inst/doc/md/
	cp $(COMMON_DIR)/doc/*.md $(SRC_DIR)/inst/doc/md/
	cd $(SRC_DIR)/vignettes/md/ && for i in *.md; do \
		pandoc -t ../../../md2rd.lua $$i | sed -e '1{/^$$/d}' | sed -e "s/^/#' /" > ../../man-roxygen/$$i.R ; \
		echo >> ../../man-roxygen/$$i.R ; \
	done
	cd $(SRC_DIR) && Rscript -e 'devtools::document(); devtools::build_vignettes()'

repository:
	cd $(PUB_DIR) && mkdir -p ./src/contrib ./bin/windows/contrib/3.2 ./bin/macosx/mavericks/contrib/3.2
	cd $(PUB_DIR) && cp -f ./$(BUNDLE_SRC) ./src/contrib/ && Rscript -e 'tools::write_PACKAGES("./src/contrib")'
	cd $(PUB_DIR) && cp -f ./$(BUNDLE_WIN) ./bin/windows/contrib/3.2/ && Rscript -e 'tools::write_PACKAGES("./bin/windows/contrib/3.2", type="win.binary")'
	cd $(PUB_DIR) && cp -f ./$(BUNDLE_MAC) ./bin/macosx/mavericks/contrib/3.2/ && Rscript -e 'tools::write_PACKAGES("./bin/macosx/mavericks/contrib/3.2", type="mac.binary")'
	## Now we cheat, since we have only limited build-system capability:
	cd $(PUB_DIR)/bin/windows/contrib && for i in 3.0 3.1 3.3 3,4; do ln -fs 3.2 $$i; done
	#cd $(PUB_DIR)/bin/windows/contrib && for i in 2.{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15} 3.0 3.1 3.3 3,4; do ln -fs 3.2 $$i; done
	cd $(PUB_DIR)/bin/macosx/mavericks/contrib && for i in 3.0 3.1 3.3 3.4; do ln -fs 3.2 $$i; done
