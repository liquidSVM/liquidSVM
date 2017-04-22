default: regenerate update

PKG_NAME=liquidSVM

MAIN_DIR=..
COMMON_DIR=$(MAIN_DIR)/bindings/common

scp=scp
WEB_HOST=isa
WEB_DIR=www/software

pack:
	cd R && $(MAKE) pack
	cd python && $(MAKE) pack
	cd java && $(MAKE) pack
	cd spark && $(MAKE) pack
	cd matlab && $(MAKE) pack-src
	#cd matlab && $(MAKE) pack

unittest:
	#cd R && $(MAKE) unittest
	#cd python && $(MAKE) unittest
	cd java && $(MAKE) unittest
	cd matlab && $(MAKE) unittest-octave
	#cd matlab && $(MAKE) unittest

$(COMMON_DIR)/doc/md/command-args-%.md: $(MAIN_DIR)/bin/%
	mkdir -p $(COMMON_DIR)/doc/md
	echo "" > $@
	$(MAIN_DIR)/bin/$< -h 1 | \
	sed '1,/------/d' | sed '1,/------/d' | \
	sed '/----/d ; /^-/ s/ /,/g' | \
	sed 's/^-\([^,]*\),\(.*\)/* `\1=c(\2)`/' | \
	sed '/^* /s/c(\([^,]*\))/\1/g' | \
	sed "/^\* /! s/^/    /" | \
	cat >> $@

$(MAIN_DIR)/%.html: $(MAIN_DIR)/%.md
	cat $< | pandoc --template $(COMMON_DIR)/doc/frag.template | \
	sed 's/\[/\&#x5b;/g' \
	> $@

pages: $(MAIN_DIR)/README.html $(MAIN_DIR)/NEWS.html
	$(scp) $(MAIN_DIR)/README.html $(MAIN_DIR)/NEWS.html $(WEB_HOST):$(WEB_DIR)/

command-args: $(COMMON_DIR)/doc/md/command-args-svm-train.md $(COMMON_DIR)/doc/md/command-args-svm-select.md $(COMMON_DIR)/doc/md/command-args-svm-test.md
