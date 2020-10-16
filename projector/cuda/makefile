objects = cudaMath.o projector.o siddon.o siddonCone.o distanceDriven3D.o
OUT = output

all: $(objects)
	cd ${OUT}; \
	nvcc -v --shared $(objects) --gpu-architecture=sm_70 -o libprojector.so

%.o: %.cu
	nvcc -O3 -v --gpu-architecture=sm_70 --compiler-options '-fPIC' -I.. -dc $< -o ${OUT}/$@

$(shell mkdir -p $(OUT))