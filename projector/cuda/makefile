objects = distanceDrivenFan.o cudaMath.o projector.o siddon.o siddonCone.o siddonFan.o distanceDriven3D.o fbp.o
OUT = output
TARGET = libprojector.so

all: $(objects)
	cd ${OUT}; \
	nvcc --shared $(objects) --gpu-architecture=sm_70 -lcufft -o ${TARGET}
	cp ${OUT}/${TARGET} ../${TARGET}

%.o: %.cu
	nvcc -O3 --gpu-architecture=sm_70 --compiler-options '-fPIC' -I.. -dc $< -o ${OUT}/$@

$(shell mkdir -p $(OUT))