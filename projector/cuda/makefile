OUT = ./output
DST = ..
TARGET = libprojector
GPUCODE_TARGET = ${OUT}/${TARGET}.gpu.o
SHARED_TARGET = ${OUT}/${TARGET}.so
STATIC_TARGET = ${OUT}/${TARGET}.a

SOURCES := $(wildcard *.cu)
OBJECTS := $(addprefix ${OUT}/,$(patsubst %.cu,%.o,$(SOURCES)))
DEPENDS := $(addprefix ${OUT}/,$(patsubst %.cu,%.d,$(SOURCES)))

.PHONY: all shared static clean

all: shared static

shared: ${SHARED_TARGET}

static: ${STATIC_TARGET}

# shared library for python
${SHARED_TARGET}: ${OBJECTS}
	nvcc --shared $(OBJECTS) -arch=sm_70 -lcufft -o ${SHARED_TARGET}
	cp ${SHARED_TARGET} ${DST}/${TARGET}.so

# static library for tensorflow
${STATIC_TARGET}: $(OBJECTS)
	nvcc -arch=sm_70 -dlink $(OBJECTS) --compiler-options '-fPIC' -o ${GPUCODE_TARGET}
	ar rcs ${STATIC_TARGET} $(OBJECTS) ${GPUCODE_TARGET}

# header dependency
-include $(DEPENDS)

# compile
${OUT}/%.o: %.cu
	nvcc -E --compiler-options '-MM -MF $(patsubst %.o,%.d,$@) -MT $@ -MP' $<
	nvcc -O3 -arch=sm_70 -I.. -dc --compiler-options '-fPIC' $< -o $@

clean: 
	-rm -f ${OBJECTS} ${STATIC_TARGET} ${SHARED_TARGET} ${GPUCODE_TARGET} ${DEPENDS}

$(shell mkdir -p $(OUT))