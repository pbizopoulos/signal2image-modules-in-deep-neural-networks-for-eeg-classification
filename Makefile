#NAME
#	Reproducible Builds for Computational Research Papers Makefile help.
#
#SEE ALSO
#	https://github.com/pbizopoulos/cookiecutter-reproducible-builds-for-computational-research-papers
#	https://github.com/pbizopoulos/reproducible-builds-for-computational-research-papers
#
#SYNTAX
#	make [OPTION] [ARGS=--full]
#
#USAGE
# 
#	+-------------------+----------------------+---------------------------+
#	|         \ ARGS    |       (empty)        |          --full           |
#	|   OPTION \        |                      |                           |
#	+-------------------+----------------------+---------------------------+
#	| (ms.pdf or empty) |  debug/development   |       release paper       |
#	+-------------------+----------------------+---------------------------+
#	|       test        | test reproducibility | test reproducibility full |
#	+-------------------+----------------------+---------------------------+
#
#OPTIONS

.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
DOCKER_GPU_ARGS=--gpus all

ms.pdf: ms.tex ms.bib results/.completed # Generate pdf.
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /home/latex/ms.tex

results/.completed: Dockerfile $(shell find . -maxdepth 1 -name '*.py')
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

test: # Test whether the paper has a reproducible build.
	make clean && make ARGS=$(ARGS) DEBUG_ARGS= DOCKER_GPU_ARGS="$(DOCKER_GPU_ARGS)" && mv ms.pdf tmp.pdf
	make clean && make ARGS=$(ARGS) DEBUG_ARGS= DOCKER_GPU_ARGS="$(DOCKER_GPU_ARGS)"
	@diff ms.pdf tmp.pdf && echo 'ms.pdf has a reproducible build.' || echo 'ms.pdf has not a reproducible build.'
	@rm tmp.pdf

clean: # Remove cache, results directories and tex auxiliary files.
	rm -rf __pycache__/ cache/* results/* results/.completed ms.bbl
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -C -cd /home/latex/ms.tex

help: # Show help.
	@grep '^#' Makefile | cut -b 2-
	@grep -E '^[a-z.-]*:.*# .*$$' Makefile | awk 'BEGIN {FS = ":.*# "}; {printf "\t%-6s - %s\n", $$1, $$2}'
