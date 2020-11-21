TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared siddonCone.cc -o libtfprojector.so -fPIC -I../cuda -L../cuda/output -lprojector ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3