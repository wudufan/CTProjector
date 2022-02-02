# objects = NonLocalMean.o TotalVariation.o
# OUT = output

# all: $(objects)
# 	cd ${OUT}; \
# 	nvcc -v --shared $(objects) --gpu-architecture=sm_70 -o libprior.so

# %.o: %.cu
# 	nvcc -O3 -v --gpu-architecture=sm_70 --compiler-options '-fPIC' -I.. -dc $< -o ${OUT}/$@

# $(shell mkdir -p $(OUT))

OUT = ./output
DST = ../../bin
TARGET = libprior
GPUCODE_TARGET = ${OUT}/${TARGET}.gpu.o
SHARED_TARGET = ${OUT}/${TARGET}.so
STATIC_TARGET = ${OUT}/${TARGET}.a

SOURCES := $(wildcard *.cu)
OBJECTS := $(addprefix ${OUT}/,$(patsubst %.cu,%.o,$(SOURCES)))
DEPENDS := $(addprefix ${OUT}/,$(patsubst %.cu,%.d,$(SOURCES)))

.PHONY: all shared clean

all: shared static

shared: ${SHARED_TARGET}

static: ${STATIC_TARGET}

# shared library for python
${SHARED_TARGET}: ${OBJECTS}
	nvcc --shared $(OBJECTS) ${GPU_CODE_FLAG} -o ${SHARED_TARGET}
	cp ${SHARED_TARGET} ${DST}/${TARGET}.so

# static library for tensorflow
${STATIC_TARGET}: $(OBJECTS)
	nvcc ${GPU_CODE_FLAG} -dlink $(OBJECTS) --compiler-options '-fPIC' -o ${GPUCODE_TARGET}
	ar rcs ${STATIC_TARGET} $(OBJECTS) ${GPUCODE_TARGET}

# header dependency
-include $(DEPENDS)

# compile
${OUT}/%.o: %.cu
	nvcc -E --compiler-options '-MM -MF $(patsubst %.o,%.d,$@) -MT $@ -MP' $<
	nvcc -O3 ${GPU_CODE_FLAG} -I.. -dc --compiler-options '-fPIC' $< -o $@

clean: 
	-rm -f ${OBJECTS} ${STATIC_TARGET} ${SHARED_TARGET} ${GPUCODE_TARGET} ${DEPENDS}

$(shell mkdir -p $(OUT))
$(shell mkdir -p $(DST))