export GPU_CODE_FLAG= \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80

SUBDIRS = prior projector

.PHONY: all cuda tensorflow ${SUBDIRS}

all: ${SUBDIRS}

${SUBDIRS}:
	$(MAKE) -C $@

cuda: $(addsuffix -cuda,${SUBDIRS})

$(addsuffix -cuda,${SUBDIRS}):
	$(MAKE) -C $(patsubst %-cuda,%,$@) cuda

tensorflow: $(addsuffix -tensorflow,${SUBDIRS})

$(addsuffix -tensorflow,${SUBDIRS}):
	$(MAKE) -C $(patsubst %-tensorflow,%,$@) tensorflow

$(addsuffix -clean,${SUBDIRS}):
	cd $(patsubst %-clean,%,$@); make clean

clean: $(addsuffix -clean,${SUBDIRS})