
default: package

THENAME=liquidSVM
PACKAGE=de.uni_stuttgart.isa.liquidsvm.spark
PACKAGE_PATH=de/uni_stuttgart/isa/liquidsvm/spark

MAIN_DIR=../..
COMMON_DIR=$(MAIN_DIR)/bindings/common

rm=\rm

## sbt-convention is to put version everything...
SCALA_VERSION=2.10

BUILD_DIR=target/scala-$(SCALA_VERSION)
BUILD_STAGE=target/scala-$(SCALA_VERSION)/stage
BUILD_JAR=$(BUILD_DIR)/$(THENAME)-spark.jar
BUILD_ZIP=$(BUILD_DIR)/$(THENAME)-spark.zip

## we need the following 
#JAVA_HOME=/usr/lib/jvm/java-8-oracle
#SCALA_HOME=$(HOME)/hd/scala-$(SCALA_VERSION)
#SPARK_HOME=$(HOME)/hd/spark-1.6
#HADOOP_HOME=$(HOME)/hd/hadoop

#SPARK_VERSION_STRING=2.11-2.1.0
#SCALAC=$(SCALA_HOME)/bin/scalac

MASTER_CONF=--master $${MASTER:-local[2]} 

## This uses scalac that is included in Spark, and hence avoids conflicts:
SCALAC=$(JAVA_HOME)/bin/java -cp $(SPARK_HOME)/jars/*:$(SPARK_HOME)/lib/* -Dscala.usejavacp=true scala.tools.nsc.Main

JAR=$(JAVA_HOME)/bin/jar

CLASSPATH_COMPILE=../java/bin/$(THENAME).jar:$(SPARK_HOME)/jars/*:$(SPARK_HOME)/lib/*:$(HADOOP_HOME)/share/common/*

OBJ_DSVM = $(BUILD_DIR)/classes/$(PACKAGE_PATH)/DistributedSVM.class
OBJ_APP = $(BUILD_DIR)/classes/$(PACKAGE_PATH)/App.class

OBJECTS = $(OBJ_DSVM) $(OBJ_APP)

clean:
	$(rm) -fr $(BUILD_DIR)/

SCALAC_CMD=mkdir -p $(BUILD_DIR)/classes/$(PACKAGE_PATH); \
	$(SCALAC) -deprecation -d $(BUILD_DIR)/classes/ -classpath $(CLASSPATH_COMPILE):$(BUILD_DIR)/classes

$(OBJ_APP): App.scala
	$(SCALAC_CMD) App.scala

$(OBJ_DSVM): App.scala
	$(SCALAC_CMD) spark.scala

compile: $(OBJECTS)
	@#$(rm) -rf $(BUILD_DIR)/classes/
	@#mkdir -p $(BUILD_DIR)/classes/$(PACKAGE_PATH)
	#make -C .. compile

package, $(BUILD_JAR): $(OBJECTS)
	$(rm) -f $(BUILD_JAR)
	## At the moment we just use the provieded ./liquidSVM.jar for the Java-Bindings
	unzip -oq ../java/bin/$(THENAME).jar -d $(BUILD_DIR)/classes/
	$(JAR) cf $(BUILD_JAR) -C $(BUILD_DIR)/classes/ .


README.html: README.md
	cat README.md | pandoc -t html --standalone > README.html

README.txt: README.md
	cat README.md | pandoc -t plain --standalone > README.txt
index.html: README.md
	@#sed 1,2d README.md | 
	cat README.md | pandoc --template $(COMMON_DIR)/doc/frag.template | sed 's/\[/\&#x5b;/g' > index.html

$(BUILD_ZIP): $(BUILD_JAR) README.html README.txt
	rm -rf $(BUILD_STAGE)
	mkdir -p $(BUILD_STAGE)
	unzip -q ../java/build/$(THENAME)-java.zip -d $(BUILD_STAGE)
	mv $(BUILD_STAGE)/$(THENAME)-java $(BUILD_STAGE)/$(THENAME)-spark
	cd $(BUILD_STAGE)/$(THENAME)-spark/ && rm $(THENAME).jar Example.java README.txt reg-1d.*.csv
	cp $(BUILD_JAR) example.scala $(BUILD_STAGE)/$(THENAME)-spark/
	cp ../../data/covtype.10000.*.csv $(BUILD_STAGE)/$(THENAME)-spark/
	cd $(BUILD_STAGE) && zip -qr ../$(THENAME)-spark.zip $(THENAME)-spark
pack: $(BUILD_ZIP)

# I'd like to use --files libliquidsvm.so but there seems to be no way to change 
# the LD_LIBRARY_PATH for the executors to include the dynamic executor directory :-|
# However, it seems that if libliquidsvm.so is in the current directory it gets loaded on the 

LIB_CONF=--conf spark.executor.extraLibraryPath=. --conf spark.driver.extraLibraryPath=. --files libliquidsvm.so

shell: $(BUILD_JAR)
	$(SPARK_HOME)/bin/spark-shell $(MASTER_CONF) --jars $(BUILD_JAR) $(LIB_CONF)

demo: $(BUILD_JAR)
	$(SPARK_HOME)/bin/spark-submit --class $(PACKAGE).App $(MASTER_CONF) $(LIB_CONF) $(BUILD_JAR) covtype.10000 20000 5000 1000


