MAIN_DIR=../..
COMMON_DIR=$(MAIN_DIR)/bindings/common
MAIN_PATH_CPP=kernel_calculator.cpp

CPPC=g++

all: kernel

prepare:
	mkdir -p bin/

config: prepare
	$(CPPC) -g -lpthread -fPIC -g -std=c++0x -lrt -o bin/config -I$(COMMON_DIR) -I$(MAIN_DIR) $(COMMON_DIR)/scenario_config.cpp -w -D COMPILE_SCENARIO_CONFIG_MAIN__

kernel: prepare
	$(CPPC) -g -lpthread -fPIC -lrt -o bin/kernel -I$(COMMON_DIR) -I$(MAIN_DIR) $(MAIN_PATH_CPP) -w -D COMPILE_KERNEL_CALCULATOR_MAIN__

main: prepare
	$(CPPC) -g -lpthread -fPIC -lrt -std=c++0x -o bin/main -I$(COMMON_DIR) -I$(MAIN_DIR) $(COMMON_DIR)/main.cpp -w -D COMPILE_FOR_COMMAND_LINE__
