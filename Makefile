#
# Matrix Multiply makefile
#
# History: Edited by Xuan Ly NGUYEN THE, August 2015

ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-msse -O3 -ffast-math

LIBS = -lm -lOpenCL -fopenmp 

COMMON_DIR = Cpp_common

INC = -I $(COMMON_DIR)

MMUL_OBJS = matmul.o matrix_lib.o wtime.o
EXEC = mult

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -lm -framework OpenCL
endif

all: $(EXEC)

mult: $(MMUL_OBJS)
	$(CPPC) $(MMUL_OBJS) $(CCFLAGS) $(LIBS) -o $(EXEC)

wtime.o: $(COMMON_DIR)/wtime.c
	$(CPPC) -c $^ $(CCFLAGS) -o $@


.c.o:
	$(CPPC) -c $< $(CCFLAGS) -o $@

.cpp.o:
	$(CPPC) -c $< $(CCFLAGS) $(INC) -o $@

matmul.o:	matmul.hpp matrix_lib.hpp

matrix_lib.o:	matmul.hpp

clean:
	rm -f $(MMUL_OBJS) $(EXEC) *.stackdump
