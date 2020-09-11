objects = cudaMath.o Projector.o Siddon.o SiddonCone.o
OUT = output

all: $(objects)
	cd ${OUT}; \
	nvcc -v --shared $(objects) -o libprojector.so

%.o: %.cu
	nvcc -O3 -v --gpu-architecture=sm_60 --compiler-options '-fPIC' -I.. -dc $< -o ${OUT}/$@

$(shell mkdir -p $(OUT))