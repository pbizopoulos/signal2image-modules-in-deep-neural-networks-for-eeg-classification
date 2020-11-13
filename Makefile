.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
VOLUME_DIR=/usr/src/app
ifeq (, $(shell which nvidia-smi))
	GPU_ARGS=
else
	GPU_ARGS=--gpus all
endif

cache/ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):$(VOLUME_DIR) \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -outdir=cache/ -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd $(VOLUME_DIR)/ms.tex
	@if [ -f cache/.tmp.pdf ]; then \
		cmp cache/ms.pdf cache/.tmp.pdf && echo 'ms.pdf unchanged.' || echo 'ms.pdf changed.'; fi
	@cp cache/ms.pdf cache/.tmp.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/*
	docker image build --tag signal2image-modules-in-deep-neural-networks-for-eeg-classification .
	docker container run \
		$(DEBUG_ARGS) \
		--env HOME=$(VOLUME_DIR)/cache \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):$(VOLUME_DIR) \
		--workdir $(VOLUME_DIR) \
		$(GPU_ARGS) \
		signal2image-modules-in-deep-neural-networks-for-eeg-classification \
		python main.py $(ARGS)
	touch results/completed

clean:
	rm -rf __pycache__/ cache/* results/*
