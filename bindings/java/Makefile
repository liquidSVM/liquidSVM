default: lib

include Makefile.pack

MAIN_DIR=../..
SRC_DIR=.
COMMON_BASE_DIR=$(MAIN_DIR)/bindings
COMMON_DIR=$(COMMON_BASE_DIR)/common
BIN_DIR=bin


PKG_NAME=liquidSVM-java
BUILD_DIR=build/$(PKG_NAME)

PUB_DIR=$(MAIN_DIR)/pub/java

rm=\rm

PATH_JAVA=src/de/uni_stuttgart/isa/liquidsvm
PATH_PACKAGE=de/uni_stuttgart/isa/liquidsvm

#MAIN_CLASS=de.uni_stuttgart.isa.liquidsvm.Util
MAIN_PATH_JAVA=src/de/uni_stuttgart/isa/liquidsvm/Util.java
MAIN_PATH_CLASS=bin/de/uni_stuttgart/isa/liquidsvm/Util.class

GUI_CLASS=de.uni_stuttgart.isa.liquidsvm.gui.SVMFrame
GUI_PATH_JAVA=src/de/uni_stuttgart/isa/liquidsvm/gui/SVMFrame.java
GUI_PATH_CLASS=bin/de/uni_stuttgart/isa/liquidsvm/gui/SVMFrame.class

LOCAL=/usr/local/isa-scratch
LOCAL_LIB_DIR=$(LOCAL)/lib

#MAIN_PATH_PREFIX=de_uni_stuttgart_isa_liquidsvm
#MAIN_PATH_CPP=$(MAIN_PATH_PREFIX)_SVM.cpp
#MAIN_PATH_H=$(MAIN_PATH_PREFIX)_SVM.h
#
#JAVA_INCLUDE_PATH=-I$(JAVA_HOME)/include/ -I$(JAVA_HOME)/include/$(OS_NAME)

#lib: $(LIBNAME)

#$(LIBNAME): $(COMMON_DIR)/liquidSVM.h $(COMMON_DIR)/liquidSVM.cpp $(MAIN_PATH_CPP)
#	mkdir -p bin
#	g++ -shared -g0 -O3 -fPIC -std=c++0x -march=native -m64 -o bin/$(LIBNAME) $(JAVA_INCLUDE_PATH) -I$(MAIN_DIR) -I$(MAIN_DIR)/bindings $(MAIN_PATH_CPP) -w

lib-local: 
	g++ -shared -g2 -O3 -fPIC -std=c++0x -lrt -march=native -o $(LOCAL_LIB_DIR)/$(LIBNAME) $(JAVA_INCLUDE_PATH) -I$(MAIN_DIR) -I$(MAIN_DIR)/bindings $(MAIN_PATH_CPP) -w

jar: compile
	cp $(PATH_JAVA)/*.java $(PATH_JAVA)/*.html bin/$(PATH_PACKAGE)
	$(JAVA_HOME)/bin/jar cfe bin/liquidSVM.jar $(MAIN_CLASS) -C bin de

gui: lib $(MAIN_PATH_CLASS) $(GUI_PATH_CLASS)
	# ##bug?## if LANG is e.g. de_ch.UTF-8 then parsing of comma-values (-s 1.0 0.001) fails!
	# ##Fixed now in cpp. Workaround was: LANG=en_US.UTF-8
	java -Djava.library.path=bin -cp bin $(GUI_CLASS)

demo2: lib $(MAIN_PATH_CLASS)
	# ##bug?## if LANG is e.g. de_ch.UTF-8 then parsing of comma-values (-s 1.0 0.001) fails!
	# ##Fixed now in cpp. Workaround was: LANG=en_US.UTF-8
	java -Djava.library.path=$(BIN_DIR) -cp $(BIN_DIR) $(MAIN_CLASS)

java_header: $(MAIN_PATH_H)

$(PATH_JAVA)/package.html index.html: overview.md $(COMMON_DIR)/doc/global-and-grid.md
	cat overview.md $(COMMON_DIR)/doc/global-and-grid.md \
	| pandoc -t html --standalone \
	> $(PATH_JAVA)/package.html
	sed 1,2d overview.md | pandoc -t plain > README.txt
	sed 1,2d overview.md | pandoc --template $(COMMON_DIR)/doc/frag.template | sed 's/\[/\&#x5b;/g' > index.html

javadoc: $(PATH_JAVA)/package.html
	mkdir -p doc
	javadoc -d doc -doctitle "liquidSVM for Java" -windowtitle "liquidSVM for Java" -quiet \
	-stylesheetfile overview.css $(PATH_JAVA)/*.java
#	rm doc/*.*
#	mv doc/$(PATH_PACKAGE)/*.* doc/
#	rm -r doc/de
#	cp doc/package-summary.html doc/index.html

$(MAIN_PATH_H): $(MAIN_PATH_CLASS)
	$(JAVA_HOME)/bin/javah -classpath bin $(NAME_PACKAGE).SVM
	rm $(MAIN_PATH_PREFIX)_SVM_*.h

compile: $(MAIN_PATH_JAVA) #$(GUI_PATH_JAVA)
	mkdir -p bin
	"$(JAVA_HOME)/bin/javac" -d bin src/de/uni_stuttgart/isa/liquidsvm/*.java

$(MAIN_PATH_CLASS) $(GUI_PATH_CLASS): compile

pack: javadoc pack-quick

pack-quick $(PKG_NAME): jar
	rm -fr $(BUILD_DIR) build/$(PKG_NAME).zip
	mkdir -p $(BUILD_DIR)
	cp -pr bin/liquidSVM.jar doc $(BUILD_DIR)
	cp -pr Makefile.pack $(BUILD_DIR)/Makefile
	cp -pr README.txt $(BUILD_DIR)
	cp -pr $(MAIN_DIR)/COPYING $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/src/common
	cp -pr $(MAIN_PATH_PREFIX)_*.* $(BUILD_DIR)/src/
	cp -pr $(COMMON_DIR)/liquidSVM.* $(COMMON_DIR)/scenario_config.* $(BUILD_DIR)/src/common/
	cp -pr $(MAIN_DIR)/sources $(BUILD_DIR)/src/
	rm -fr $(BUILD_DIR)/src/sources/*/object_code/*
	cp -p bin/$(LIBNAME_LINUX) bin/$(LIBNAME_MAC) bin/$(LIBNAME_WIN) $(BUILD_DIR) 2>/dev/null || :
	cp src/Example.java $(BUILD_DIR)/Example.java
	cp $(MAIN_DIR)/data/reg-1d.t*.csv $(BUILD_DIR)
	cd build && zip -qr $(PKG_NAME).zip $(PKG_NAME)
#	rm -r $(BUILD_DIR) 

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
		ssh -x $(SSH_WIN) 'cd $(SSH_WIN_DIR); rm -r $(PKG_NAME); unzip -q $(PKG_NAME).zip; cd $(PKG_NAME); make generic'
		scp $(SSH_WIN):$(SSH_WIN_DIR)/$(PKG_NAME)/$(LIBNAME_WIN) bin/$(LIBNAME_WIN)

pub: $(PKG_NAME) $(SRC_PKG_NAME) pub-quick

pub-quick: index.html pack-quick javadoc
	mkdir -p $(PUB_DIR)
	cp index.html build/$(PKG_NAME).zip $(PUB_DIR)
	$(rm) -r $(PUB_DIR)/doc 2>/dev/null || :
	cp -r doc $(PUB_DIR)


ECLIPSE_HOME=$(HOME)/opt/eclipse
JUNIT=$(ECLIPSE_HOME)/plugins/org.junit_4.12.0.v201504281640/junit.jar
HAMCREST=$(ECLIPSE_HOME)/plugins/org.hamcrest.core_1.3.0.v201303031735.jar
JUNIT_CLASSPATH=$(JUNIT):$(HAMCREST)
JUNIT_RUNNER=org.junit.runner.JUnitCore
UNITTEST_CLASS=de.uni_stuttgart.isa.liquidsvm.AllTests

unittest: #compile
	mkdir -p bin/junit
	$(JAVA_HOME)/bin/javac -classpath bin:$(JUNIT_CLASSPATH) -d bin/junit junit/de/uni_stuttgart/isa/liquidsvm/*.java
	java -Djava.library.path=bin -cp bin:bin/junit:$(JUNIT_CLASSPATH) $(JUNIT_RUNNER) $(UNITTEST_CLASS)

clean:
	$(rm) -rf bin/*

