.POSIX:

ARGS= 
CACHE_DIR=cache
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
PAPER_TITLE=signal2image-modules-in-deep-neural-networks-for-eeg-classification
RESULTS_DIR=results
VOLUME_DIR=/usr/src/app

ifeq (, $(shell which nvidia-smi))
	GPU_ARGS=
else
	GPU_ARGS=--gpus all
endif

$(CACHE_DIR)/ms.pdf: ms.tex ms.bib $(RESULTS_DIR)/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):$(VOLUME_DIR) \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -outdir=$(CACHE_DIR)/ -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd $(VOLUME_DIR)/ms.tex
	@if [ -f $(CACHE_DIR)/.tmp.pdf ]; then \
		cmp $(CACHE_DIR)/ms.pdf $(CACHE_DIR)/.tmp.pdf && echo 'ms.pdf unchanged.' || echo 'ms.pdf changed.'; fi
	@cp $(CACHE_DIR)/ms.pdf $(CACHE_DIR)/.tmp.pdf

$(RESULTS_DIR)/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf $(RESULTS_DIR)/*
	docker image build --tag $(PAPER_TITLE) .
	docker container run \
		$(DEBUG_ARGS) \
		--env HOME=$(VOLUME_DIR)/$(CACHE_DIR)/ \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):$(VOLUME_DIR) \
		--workdir $(VOLUME_DIR) \
		$(GPU_ARGS) \
		$(PAPER_TITLE) \
		python main.py $(CACHE_DIR)/ $(RESULTS_DIR)/ $(ARGS)
	touch $(RESULTS_DIR)/completed

clean:
	rm -rf __pycache__/ $(CACHE_DIR)/* $(RESULTS_DIR)/*
