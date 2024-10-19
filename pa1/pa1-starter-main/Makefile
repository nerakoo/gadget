# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the MY_OPT variable to invoke additional compiler options

#########################################################################
#									#
# Sample makefile header for running with Gnu compilers  		#
# Both MPI and OMP compilation may be independently disabled		#
#									#
#  The makefile targets are appended to  the end of this file		#
#	 Don't change anything that comes before the targets 		#
#									#
#									#
#########################################################################

STDCPP          = --std=c11
STDC            = --std=c11

C++ 		= g++ $(STDCPP)
CC 		= gcc $(STDC)
# C++		= clang++
# CC		= clang 

# include openblas library
LIB_BLAS =  -lopenblas
INCLUDES += -I/usr/include/openblas
# and ARM Performance Libarary
ARMPL = -L/opt/arm/armpl_22.0.2_gcc-11.2/lib -larmpl

LDLIBS += $(LIB_BLAS) $(ARMPL)


# This generates output about how the
# compiler vectorized the code
# We  suggest using level 2 (the integer after "verbose=")
# See the gcc manual for the other levels of output: levels 1-7
# http://gcc.gnu.org/onlinedocs/gcc-4.7.3/gcc/Debugging-Options.html#Debugging-Options
# REPORT          = -ftree-vectorizer-verbose=2
# OPTIMIZATION += -ftree-vectorize


# ARCH_FLAGS      =  -m64
WARNINGS        = 
OPTIMIZATION    =  
ifeq ($(omp),1)
OPTIMIZATION    +=  -fopenmp
endif

C++FLAGS        += $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG) $(REPORT)

CFLAGS		+= $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG) $(REPORT)

LDLIBS		+= -lm -pthread


## Original Make file

CFLAGS += "-g"
CFLAGS += "-DOPENBLAS_SINGLETHREAD"
CFLAGS += -march=armv8.4-a+sve
CFLAGS += -msve-vector-bits=256

# If you want to change your optimization settings, do it here.
ifeq ($(debug), 1)
	MY_OPT = "-O0"
else
	MY_OPT = "-O4"
endif
# MY_OPT += "-fPIC"
# MY_OPT = "-O3"
# MY_OPT = "-O4"

OPTIMIZATION += $(MY_OPT)

# WARNINGS += -Wall -pedantic
WARNINGS += -Wall 
# WARNINGS += -w

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
endif

C++FLAGS += -DNO_BLAS
CFLAGS += -DNO_BLAS

targets = benchmark-naive benchmark-blas benchmark-blislab
BLISLAB = blislab/bl_dgemm_ukr.c  blislab/my_dgemm.c blislab/bl_dgemm_util.c
#	blislab/bl_dgemm_asm_4x12.c \
#	blislab/bl_dgemm_asm_8x4.c
objects = benchmark.o \
	bl_dgemm_util.o \
	blas/dgemm-blas.o \
	naive/dgemm-naive.o \
	$(BLISLAB:.c=.o)

UTIL   = wall_time.o cmdLine.o debugMat.o blislab/bl_dgemm_util.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o naive/dgemm-naive.o  $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -static
benchmark-blas : benchmark.o blas/dgemm-blas.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -static
benchmark-blislab : $(BLISLAB:.c=.o) benchmark.o $(UTIL) 
	$(CC) -o $@ $^ $(LDLIBS) -pg -static

%.o : %.c
	$(CC) -c $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(C++) $(C++FLAGS) -c $< -o $@

.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core

