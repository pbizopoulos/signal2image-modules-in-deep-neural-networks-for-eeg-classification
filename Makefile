.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
ifeq (, $(shell which nvidia-smi))
	DOCKER_GPU_ARGS=
else
	DOCKER_GPU_ARGS=--gpus all
endif

ms.pdf: ms.tex ms.bib results/.completed
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /home/latex/ms.tex

results/.completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/* results/.completed
	docker image build --tag signal2image-modules-in-deep-neural-networks-for-eeg-classification .
	docker container run \
		$(DEBUG_ARGS) \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		$(DOCKER_GPU_ARGS) signal2image-modules-in-deep-neural-networks-for-eeg-classification \
		python main.py $(ARGS)
	touch results/.completed

test:
	make clean && make ARGS=$(ARGS) DEBUG_ARGS= && mv ms.pdf tmp.pdf
	make clean && make ARGS=$(ARGS) DEBUG_ARGS=
	@diff ms.pdf tmp.pdf && echo 'ms.pdf has a reproducible build.' || echo 'ms.pdf has not a reproducible build.'
	@rm tmp.pdf

clean:
	rm -rf __pycache__/ cache/* results/* results/.completed ms.bbl
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -C -cd /home/latex/ms.tex
