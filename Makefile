.POSIX:

ARGS= 
PAPER_TITLE=signal2image-modules-in-deep-neural-networks-for-eeg-classification

ifeq (1, $(shell [ -t 0 ] && echo 1))
	DEBUG_ARGS=--interactive --tty
endif

ifneq (, $(shell which nvidia-smi))
	GPU_ARGS=--gpus all
endif

tmp/ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/usr/src/app \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -outdir=tmp/ -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /usr/src/app/ms.tex
	@if [ -f tmp/.ms-latest.pdf ]; then \
		cmp tmp/ms.pdf tmp/.ms-latest.pdf && echo 'tmp/ms.pdf unchanged.' || echo 'tmp/ms.pdf changed.'; fi
	@cp tmp/ms.pdf tmp/ms-`date --iso-8601=seconds`.pdf
	@cp tmp/ms.pdf tmp/.ms-latest.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results
	mkdir -p results
	mkdir -p tmp
	docker image build --tag $(PAPER_TITLE) .
	docker container run \
		$(DEBUG_ARGS) \
		--env HOME=/usr/src/app/tmp/ \
		--detach-keys "ctrl-^,ctrl-^"  \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/usr/src/app \
		--workdir /usr/src/app \
		$(GPU_ARGS) \
		$(PAPER_TITLE) \
		python main.py results/ tmp/ $(ARGS)
	touch results/completed

clean:
	rm -rf results/ tmp/
