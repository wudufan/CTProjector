objects = siddonCone.o
OUT = output
TARGET = libtfprojector.so
TF_INCLUDE ?= $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LFLAGS ?= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all: $(objects)
	cd ${OUT}; \
	g++ $(objects) -shared -L../../cuda/output -lprojector ${TF_LFLAGS} -o ${TARGET}
	cp ${OUT}/${TARGET} ../${TARGET}

%.o: %.cc
	g++ -c -std=c++11 -O3 -fPIC -I../cuda -I${TF_INCLUDE} -D_GLIBCXX_USE_CXX11_ABI=1 $< -o ${OUT}/$@

$(shell mkdir -p $(OUT))