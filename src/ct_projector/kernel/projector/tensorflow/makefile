OUT = ./output
DST = ../../bin
TARGET = libtfprojector

SHARED_TARGET = ${OUT}/${TARGET}.so

SOURCES := $(wildcard *.cpp)
OBJECTS := $(addprefix ${OUT}/,$(patsubst %.cpp,%.o,$(SOURCES)))
DEPENDS := $(addprefix ${OUT}/,$(patsubst %.cpp,%.d,$(SOURCES)))

LFLAGS=-L../cuda/output -l:libprojector.a -lcufft
CFLAGS=-I../cuda

TF_INCLUDE ?= $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LFLAGS ?= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

.PHONY: all shared clean

all: shared

shared: ${SHARED_TARGET}

# shared library
${SHARED_TARGET}: ${OBJECTS}
	nvcc $(OBJECTS) -shared ${LFLAGS} ${TF_LFLAGS} -o ${SHARED_TARGET}
	cp ${SHARED_TARGET} ${DST}/${TARGET}.so

# dependency
-include $(DEPENDS)

#compile
${OUT}/%.o: %.cpp
	nvcc -E ${CFLAGS} -I${TF_INCLUDE} --compiler-options '-MM -MF $(patsubst %.o,%.d,$@) -MT $@ -MP' $<
	nvcc -c -O3 ${CFLAGS} -I${TF_INCLUDE} --compiler-options "-fPIC -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1" $< -o $@

clean: 
	-rm -f ${OBJECTS} ${SHARED_TARGET} ${DEPENDS}

$(shell mkdir -p $(OUT))
$(shell mkdir -p $(DST))