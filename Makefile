.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
ifeq (, $(shell which nvidia-smi))
	DOCKER_GPU_ARGS=
else
	DOCKER_GPU_ARGS=--gpus all
endif

ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		ghcr.io/pbizopoulos/texlive-full \
		-usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /usr/src/app/ms.tex
	@if [ -f cache/.tmp.pdf ]; then \
		cmp ms.pdf cache/.tmp.pdf && echo 'ms.pdf unchanged.' || echo 'ms.pdf changed.'; fi
	@cp ms.pdf cache/.tmp.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/*
	docker image build --tag signal2image-modules-in-deep-neural-networks-for-eeg-classification .
	docker container run \
		$(DEBUG_ARGS) \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		$(DOCKER_GPU_ARGS) \
		signal2image-modules-in-deep-neural-networks-for-eeg-classification \
		python main.py $(ARGS)
	touch results/completed

clean:
	rm -rf __pycache__/ cache/* results/* ms.bbl
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		ghcr.io/pbizopoulos/texlive-full \
		-C -cd /usr/src/app/ms.tex
