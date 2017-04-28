#----------------------------------------------------------------------------------------------------------------
#----------- Paths ----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


#----------- Generalities -----------------------------------------------------------------------------
# The next line gets the name of the current OS. It will be used later.

ifeq ($(OS),Windows_NT)
	OS_NAME= win32
else
	OS_NAME= $(shell uname -s)
endif


#----------- Target paths and files -----------------------------------------------------------------------------
# For public packages, only the first line is necessary. The rest is for development, only.
# These values should not be changed.

BIN_PATH = ./bin
BASE_SAVE_NAME= liquidSVM
A_SAVE_NAME= $(BASE_SAVE_NAME)-$(shell date -I)
R_SAVE_NAME= $(BASE_SAVE_NAME)-for-r
P_SAVE_NAME= $(BASE_SAVE_NAME)


#----------- CUDA-related paths ---------------------------------------------------------------------------------
# If you have CUDA installed, but your paths are different, then these lines need to be
# adapted.

NVCC_PATH= /usr/local/cuda/bin
CUDA_INCLUDE_PATH= /usr/local/cuda/include
CUDA_LIB_PATH= /usr/local/cuda/lib64

NVCC= $(NVCC_PATH)/nvcc


#----------------------------------------------------------------------------------------------------------------
#----------- Compiler flags -------- ----------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


#----------- GCC/C++ related flags -------------------------------------------------------

# Determine local g++ version at your computer if there is one. This is useful, if the OS does 
# not provide reasonably new version of gcc, but such a version is installed manually somewhere. 
# All what needs to be done, is to change the value of LOCAL_CPPC_PATH. If at this path there
# is no gcc, then the standard g++ will be used.

LOCAL_CPPC_PATH= /opt/gcc-4.9.0
LOCAL_CPPC= $(LOCAL_CPPC_PATH)/bin/g++
LOCAL_CPPC_EXIST_FLAG= $(strip $(notdir $(wildcard $(LOCAL_CPPC))))
CPPC= $(strip $(if $(LOCAL_CPPC_EXIST_FLAG), $(LOCAL_CPPC), g++))
LD_LINKER_FLAGS= $(if $(LOCAL_CPPC_EXIST_FLAG), -L$(LOCAL_CPPC_PATH)/lib64/lib, )    
INCLUDE_PATH= -I.


# Set compile options. First check, if version is private development version or an
# official, public version.

DEVELOP_FILE= ./.dev
DEVELOP_FLAG= $(strip $(notdir $(wildcard $(DEVELOP_FILE))))
DEVELOP_OPT= $(if $(DEVELOP_FLAG), -DOWN_DEVELOP__, -UOWN_DEVELOP__) 

PRO_VERSION_FILES= ./sources/svm/solver/*.pro.h
PRO_VERSION_FLAG= $(strip $(notdir $(wildcard $(PRO_VERSION_FILES))))
PRO_VERSION_OPT= $(if $(PRO_VERSION_FLAG), -DPRO_VERSION__, -UPRO_VERSION__) 


# Now we check if for an old version of gcc we need to compile without C++11 standard
# (or even without any c++-standard flag set for stone-age gcc versions)
 
ifeq ($(CPPC), g++)
	GCCVERSION= $(shell gcc --version | grep ^gcc | sed 's/^.* //g')
endif

ifneq ($(GCCVERSION),)
	GCCNEW= $(shell expr '$(GCCVERSION)' \>= 4.8.0)
	GCCOLD= $(shell expr '$(GCCVERSION)' \< 4.5.0)
	ifeq ($(GCCNEW), 1)
		CPP_VERSION= -std=c++11
	else
		CPP_VERSION= -std=c++0x
	endif
	ifeq ($(GCCOLD), 1)
		CPP_VERSION=
	endif
else
	CPP_VERSION= -std=c++11
	GCCVERSION='--- no version detection run ---'
endif


# Next, we define the optimization flags etc for gcc

ARM_FILE= ./.arm
ARM_FLAG= $(strip $(notdir $(wildcard $(ARM_FILE))))
MACHINE_FLAGS= $(if $(ARM_FLAG), -mcpu=cortex-a53 -mfpu=neon-vfpv4, -march=native) 
MATH_FLAGS= -ffast-math
OPT_FLAGS= -O3 
WARN_FLAGS= -Wall

MKDIR_P=mkdir -p

# Now, linking requires different parameters for different OSs

ifeq ($(OS_NAME), Linux)
	LINK_FLAGS= -g0 -lpthread -lrt -fPIC
endif
ifeq ($(OS_NAME), Darwin)
	LINK_FLAGS= -g0 -fPIC 
endif


# Finally, we glue everything together

CPP_FLAGS= $(MACHINE_FLAGS) $(MATH_FLAGS) $(OPT_FLAGS) $(CPP_VERSION) $(INCLUDE_PATH) $(LINK_FLAGS) $(WARN_FLAGS) $(DEVELOP_OPT) $(PRO_VERSION_OPT) -DCOMPILE_SEPERATELY__ -DCOMPILE_WITHOUT_EXCEPTIONS__ -DCOMPILE_FOR_COMMAND_LINE__ $(LD_LINKER_FLAGS)


#----------- CUDA-compiler related flags --------------------------------------------------
# The architecure flag can be set to higher values, if the hardware supports this.
# The value below should be safe for essentially all non stone-age systems.

NVCCFLAGS= -arch sm_30 -L$(CUDA_LIB_PATH) $(INCLUDE_PATH) -DCOMPILE_WITH_CUDA__ -DCOMPILE_SEPERATELY__ -DCOMPILE_WITHOUT_EXCEPTIONS__ -U__SSE2__ -U__AVX__ 


#----------- CUDA related flags for GCC ----------------------------------------------------
# These are only activated, if CUDA-compiler NVCC is detected.

CPP_CUDA_LINK_FLAGS= -L$(CUDA_LIB_PATH) -static-libstdc++ -lcuda -lcudart
CPP_CUDA_INCLUDE_FLAGS= -I$(CUDA_INCLUDE_PATH)
NVCC_EXIST_FLAG= $(strip $(notdir $(wildcard $(NVCC))))
CPP_CUDA_FLAGS= $(if $(NVCC_EXIST_FLAG), -fPIC -DCOMPILE_WITH_CUDA__ $(DEVELOP_OPT) $(CPP_CUDA_LINK_FLAGS) $(CPP_CUDA_INCLUDE_FLAGS),)     


#----------- All GCC flags --------------------------------------------------------------------------------------

CPP_ALL_FLAGS= $(CPP_FLAGS) $(CPP_CUDA_FLAGS) 




#----------------------------------------------------------------------------------------------------------------
#----------- Code and objects -----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


#----------- Shared code and objects ----------------------------------------------------------------------------
# First we identify the source and header files that are used for the svm code and the tools.

SHARED_INLINE_SOURCES= $(wildcard ./sources/shared/*/*.ins.cpp)
SHARED_SOURCES= $(filter-out $(SHARED_INLINE_SOURCES), $(wildcard ./sources/shared/*/*.cpp))
SHARED_HEADERS= $(wildcard ./sources/shared/*/*.h)
SHARED_OBJECTS= $(patsubst %,./sources/shared/object_code/%, $(notdir $(SHARED_SOURCES:.cpp=.o)))


#----------- Code for tools ----------------------------------------------------------------------------
# ... and here we identify the corresponding files for the tools.

TOOLS_SOURCES= $(wildcard ./sources/tools/*.cpp)
TOOLS_NAMES= $(basename $(notdir $(TOOLS_SOURCES)))


#----------- CUDA code and objects ---------------------------------------------------------
# Some files need to be compiled by NVCC if CUDA is present. Here we identify these files.

CUDA_SHARED_INLINE_SOURCES= $(wildcard ./sources/shared/*/*.ins.cu)
CUDA_SHARED_SOURCES= $(if $(NVCC_EXIST_FLAG), $(filter-out $(CUDA_SHARED_INLINE_SOURCES), $(wildcard ./sources/shared/*/*.cu)),)
CUDA_SHARED_HEADERS= $(patsubst %.cu, %.h, $(CUDA_SHARED_SOURCES))
CUDA_SHARED_OBJECTS= $(patsubst %,./sources/shared/object_code/%, $(notdir $(CUDA_SHARED_SOURCES:.cu=.o)))

CUDA_SVM_INLINE_SOURCES= $(wildcard ./sources/svm/*/*.ins.cu)
CUDA_SVM_SOURCES= $(if $(NVCC_EXIST_FLAG), $(filter-out $(CUDA_SVM_INLINE_SOURCES), $(wildcard ./sources/svm/*/*.cu)),)
CUDA_SVM_HEADERS= $(patsubst %.cu, %.h, $(CUDA_SVM_SOURCES))
CUDA_SVM_OBJECTS= $(patsubst %,./sources/svm/object_code/%, $(notdir $(CUDA_SVM_SOURCES:.cu=.o)))

CUDA_OBJECTS= $(CUDA_SHARED_OBJECTS) $(CUDA_SVM_OBJECTS) 


#----------- SVM specific code and objects -----------------------------------------------------------------------
# This section identifies the source and header files for SVM specific code.

SVM_INLINE_SOURCES= $(wildcard ./sources/svm/*/*.ins.cpp)
SVM_SOURCES= $(filter-out $(SVM_INLINE_SOURCES), $(filter-out $(wildcard ./sources/svm/main/*.cpp), $(wildcard ./sources/svm/*/*.cpp)))
SVM_HEADERS= $(wildcard ./sources/svm/*/*.h)
SVM_OBJECTS= $(patsubst %,./sources/svm/object_code/%, $(notdir $(SVM_SOURCES:.cpp=.o)))

SVM_MAINS= $(wildcard ./sources/svm/main/*.cpp)
SVM_MAINS_NAMES= $(basename $(notdir $(SVM_MAINS)))


#----------- Cluster specific code and objects -----------------------------------------------------------------------
# And here we look for code that is related to cluster algorithms. This is currently for development purposes, only.

CLUSTER_INLINE_SOURCES= $(wildcard ./sources/cluster/*/*.ins.cpp)
CLUSTER_SOURCES= $(filter-out $(CLUSTER_INLINE_SOURCES), $(filter-out $(wildcard ./sources/cluster/main/*.cpp), $(wildcard ./sources/cluster/*/*.cpp)))
CLUSTER_HEADERS= $(wildcard ./sources/cluster/*/*.h)
CLUSTER_OBJECTS= $(patsubst %,./sources/cluster/object_code/%, $(notdir $(CLUSTER_SOURCES:.cpp=.o)))

CLUSTER_MAINS= $(wildcard ./sources/cluster/main/*.cpp)
CLUSTER_MAINS_NAMES= $(basename $(notdir $(CLUSTER_MAINS)))


#----------- Combine all objects and set make search paths -------------------------------------------------------
# Finally, we put everything together.

vpath %.cpp $(sort $(dir $(SHARED_SOURCES))) $(sort $(dir $(SVM_SOURCES))) $(sort $(dir $(CLUSTER_SOURCES)))
vpath %.cu $(sort $(dir $(CUDA_SHARED_SOURCES))) $(sort $(dir $(CUDA_SVM_SOURCES)))
vpath %.h $(sort $(dir $(SHARED_HEADERS))) $(sort $(dir $(SVM_HEADERS))) $(sort $(dir $(CLUSTER_HEADERS)))



#----------------------------------------------------------------------------------------------------------------
#----------- Target/Source relations ----------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


all: clean svm-all tools
svm-all: $(SVM_MAINS_NAMES)
	@if [ "$(OS_NAME)" != "win32" ]; then find . -type f -name "*.sh" -exec chmod a+x {} \; ; fi
	@$(MKDIR_P) results
tools: $(TOOLS_NAMES)
extra:
	@echo Moving additional python scripts to ./scripts
	@rm -f ./scripts/*.py
	@cp ./sources/extra/*.py ./scripts/
all-packages: win-package-sse2 win-package-avx win-package-avx2 ix-package
info: 
	@echo 
	@echo -----------------------------------------------------------------
	@echo 
	@echo The compiler is $(CPPC) version: $(GCCVERSION)
	@echo The following C++ standard flag is set: $(CPP_VERSION)
	@echo 
	@echo -----------------------------------------------------------------
	@echo


#----------- C++ related Target/Sources --------------------------------------------------------------------------------------

$(SVM_MAINS_NAMES): $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(SVM_OBJECTS) $(CUDA_SVM_OBJECTS)
	@echo Compiling and linking $(notdir $@)
	@$(MKDIR_P) $(BIN_PATH)
	@$(CPPC)  ./sources/svm/main/$@.cpp $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(SVM_OBJECTS) $(CUDA_SVM_OBJECTS) $(CPP_ALL_FLAGS) -o $(BIN_PATH)/$@

	
$(CLUSTER_MAINS_NAMES): $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(CLUSTER_OBJECTS)
	@echo Compiling and linking $(notdir $@) 
	@$(MKDIR_P) $(BIN_PATH)
	@$(CPPC)  ./sources/cluster/main/$@.cpp $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(CLUSTER_OBJECTS) $(CPP_ALL_FLAGS) -o $(BIN_PATH)/$@
	
	
$(TOOLS_NAMES): $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(SVM_OBJECTS) $(CUDA_SVM_OBJECTS) 
	@echo Compiling and linking $(notdir $@)
	@$(MKDIR_P) $(BIN_PATH)
	@$(CPPC)  ./sources/tools/$@.cpp $(SHARED_OBJECTS) $(CUDA_SHARED_OBJECTS) $(SVM_OBJECTS) $(CUDA_SVM_OBJECTS)  $(CPP_ALL_FLAGS) -o $(BIN_PATH)/$@


	
$(SHARED_OBJECTS): ./sources/shared/object_code/%.o : %.cpp $(SHARED_HEADERS) $(SHARED_INLINE_SOURCES)
	@echo Creating shared/$(notdir $@)
	@$(MKDIR_P) $(@D)
	@$(CPPC) -c ./sources/shared/*/$(patsubst %.o,%.cpp, $(notdir $@)) $(CPP_ALL_FLAGS) -o $@

	
$(SVM_OBJECTS): ./sources/svm/object_code/%.o : %.cpp $(SHARED_HEADERS) $(SHARED_INLINE_SOURCES) $(SVM_HEADERS) $(SVM_INLINE_SOURCES)
	@echo Creating svm/$(notdir $@)
	@$(MKDIR_P) $(@D)
	@$(CPPC) -c ./sources/svm/*/$(patsubst %.o,%.cpp, $(notdir $@)) $(CPP_ALL_FLAGS) -o $@

	
$(CLUSTER_OBJECTS): ./sources/cluster/object_code/%.o : %.cpp $(SHARED_HEADERS) $(SHARED_INLINE_SOURCES) $(CLUSTER_HEADERS) $(CLUSTER_INLINE_SOURCES)
	@echo Creating cluster/$(notdir $@)
	@$(MKDIR_P) $(@D)
	@$(CPPC) -c ./sources/cluster/*/$(patsubst %.o,%.cpp, $(notdir $@)) $(CPP_ALL_FLAGS) -o $@

	
#----------- CUDA related Target/Sources --------------------------------------------------------------------------------------

$(CUDA_SHARED_OBJECTS): ./sources/shared/object_code/%.o : %.cu $(CUDA_SHARED_HEADERS) $(SHARED_HEADERS) $(SHARED_INLINE_SOURCES)
ifneq ($(NVCC_EXIST_FLAG),)
	@echo $(notdir $(NVCC)): creating $(notdir $@)
	@$(MKDIR_P) $(@D)
	@$(NVCC) $(CPP_CUDA_INCLUDE_FLAGS) $(NVCCFLAGS) -c ./sources/shared/*/$(patsubst %.o,%.cu, $(notdir $@)) -o $@
else
	@echo Cannot compile $@ with $(notdir $(NVCC))
endif


$(CUDA_SVM_OBJECTS): ./sources/svm/object_code/%.o : %.cu $(CUDA_SVM_HEADERS) $(SHARED_HEADERS) $(SHARED_INLINE_SOURCES) $(SVM_HEADERS) $(SVM_INLINE_SOURCES)
ifneq ($(NVCC_EXIST_FLAG),)
	@echo $(notdir $(NVCC)): creating $(notdir $@)
	@$(MKDIR_P) $(@D)
	@$(NVCC) $(CPP_CUDA_INCLUDE_FLAGS) $(NVCCFLAGS) -c ./sources/svm/*/$(patsubst %.o,%.cu, $(notdir $@)) -o $@
else
	@echo Cannot compile $@ with $(notdir $(NVCC))
endif





#----------- Target/Sources for cleaning up and packaging ------------------------------------------------------------------------



clean: 
	@echo Removing all backups
	@find . -type f -name "*.bak" -exec rm -f {} \;
	@find . -type f -name "*.backup" -exec rm -f {} \;
	@find . -type f -name "*~" -exec rm -f {} \;
	
	@echo Removing all training files
	@find results -type f -name "*.log" -exec rm -f {} \;
	@find results -type f -name "*.sol" -exec rm -f {} \;
	@find results -type f -name "*.aux" -exec rm -f {} \;
	@find results -type f -name "*.dkr" -exec rm -f {} \;
	@find results -type f -name "*.result.*" -exec rm -f {} \;
	@find results -type f -name "*.prototxt" -exec rm -f {} \;
	
	@echo Removing all object_code and binaries
	@rm -f ./bin/*
	@find . -type f -name "*.o" -exec rm -f {} \;	


cleaner: clean
	@echo Removing training files that were accidentically saved somewhere
	@find . -type f -name "*.log" -exec rm -f {} \;
	@find . -type f -name "*.sol" -exec rm -f {} \;
	@find . -type f -name "*.aux" -exec rm -f {} \;
	@find . -type f -name "*.dkr" -exec rm -f {} \;
	@find . -type f -name "*.result.*" -exec rm -f {} \;
	@find . -type f -name "*.prototxt" -exec rm -f {} \;

	
# The next four commands are for development, only. They will not work in the final public versions.	
	

help:
	@echo
	@echo "make <target>"
	@echo "make [VERSION=full|pro|dev|public] archive|r-package|ix-package|win-package-sse2|win-package-avx|win-package-avx2"
	@echo
	
archive: cleaner
	@echo Packing all code into ../$(A_SAVE_NAME).gz
	@rm -f ../$(A_SAVE_NAME).gz
	@chmod u+x ./develop/*.sh
	@./develop/create-package.sh ~ $(A_SAVE_NAME) a full

r-package: cleaner
	@echo Packing all code into ../$(R_SAVE_NAME).gz
	@rm -f ../$(R_SAVE_NAME).gz
	@chmod u+x ./develop/*.sh
	@echo $(VERSION)
	@./develop/create-package.sh ~ $(R_SAVE_NAME) r $(VERSION)
	
ix-package: cleaner
	@echo Packing all code into ../$(P_SAVE_NAME).gz
	@rm -f ../$(P_SAVE_NAME).gz
	@chmod u+x ./develop/*.sh
	@./develop/create-package.sh ~ $(P_SAVE_NAME) c $(VERSION)
	
win-package-sse2: cleaner
	@echo Packing all code into ../$(P_SAVE_NAME)-sse2.zip
	@rm -f ../$(P_SAVE_NAME)-sse2.zip
	@chmod u+x ./develop/*.sh
	@./develop/create-package.sh ~ $(P_SAVE_NAME)-sse2 w $(VERSION) sse2
	
win-package-avx: cleaner
	@echo Packing all code into ../$(P_SAVE_NAME)-avx.zip
	@rm -f ../$(P_SAVE_NAME)-avx.zip
	@chmod u+x ./develop/*.sh
	@./develop/create-package.sh ~ $(P_SAVE_NAME)-avx w $(VERSION) avx

win-package-avx2: cleaner
	@echo Packing all code into ../$(P_SAVE_NAME)-avx2.zip
	@rm -f ../$(P_SAVE_NAME)-avx2.zip
	@chmod u+x ./develop/*.sh
	@./develop/create-package.sh ~ $(P_SAVE_NAME)-avx2 w $(VERSION) avx2
