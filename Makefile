# NAME
# 	Reproducible builds for computational research papers Makefile help.
# 
# SYNTAX
# 	make [OPTION] [ARGS=--full]
# 
# OPTIONS

.POSIX:

ARGS=
GPU=--gpus all
INTERACTIVE=-it

ms.pdf: ms.tex ms.bib results/.completed # Generate pdf.
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /home/latex/ms.tex


results/.completed: Dockerfile $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/* results/.completed
	docker build -t signal2image-modules-in-deep-neural-networks-for-eeg-classification .
	docker run --rm $(INTERACTIVE) \
		--user $(shell id -u):$(shell id -g) \
		-w /usr/src/app \
		-e HOME=/usr/src/app/cache \
		-v $(PWD):/usr/src/app \
		 $(GPU)  signal2image-modules-in-deep-neural-networks-for-eeg-classification \
		python3 main.py $(ARGS) --cache-dir cache --results-dir results
	touch results/.completed

test: # Test whether the paper has a deterministic build.
	make clean && make ARGS=$(ARGS) GPU="$(GPU)" INTERACTIVE= && mv ms.pdf tmp.pdf
	make clean && make ARGS=$(ARGS) GPU="$(GPU)" INTERACTIVE= 
	@diff ms.pdf tmp.pdf && echo 'ms.pdf has a deterministic build.' || echo 'ms.pdf has not a deterministic build.'
	@rm tmp.pdf

clean: # Remove cache, results directories and tex auxiliary files.
	rm -rf __pycache__/ cache/* results/* results/.completed ms.bbl
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -C -cd /home/latex/ms.tex

help: # Show help.
	@grep '^# ' Makefile | cut -b 3-
	@grep -E '^[a-z.-]*:.*# .*$$' Makefile | awk 'BEGIN {FS = ":.*# "}; {printf "\t%-18s - %s\n", $$1, $$2}'
