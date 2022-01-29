SUBDIRS = cuda tensorflow

.PHONY: all ${SUBDIRS}

all: ${SUBDIRS}

${SUBDIRS}:
	$(MAKE) -C $@

tensorflow: cuda
	$(MAKE) -C $@

$(addsuffix -clean,${SUBDIRS}):
	cd $(patsubst %-clean,%,$@); make clean

clean: $(addsuffix -clean,${SUBDIRS})