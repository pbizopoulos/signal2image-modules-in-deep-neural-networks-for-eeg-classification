.POSIX:

ARGS= 
PAPER_TITLE=signal2image-modules-in-deep-neural-networks-for-eeg-classification

ifeq (1, $(shell [ -t 0 ] && echo 1))
	DEBUG_ARGS=--interactive --tty
endif

ifneq (, $(shell which nvidia-smi))
	GPU_ARGS=--gpus all
endif

tmp/ms.pdf: ms.tex ms.bib tmp/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -outdir=tmp -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /workspace/ms.tex
	@if [ -f tmp/.ms-latest.pdf ]; then \
		cmp tmp/ms.pdf tmp/.ms-latest.pdf && echo 'tmp/ms.pdf unchanged.' || echo 'tmp/ms.pdf changed.'; fi
	@cp tmp/ms.pdf tmp/ms-`date --iso-8601=seconds`.pdf
	@cp tmp/ms.pdf tmp/.ms-latest.pdf

tmp/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	mkdir -p tmp
	docker image build --tag $(PAPER_TITLE) .
	docker container run \
		$(DEBUG_ARGS) \
		--env HOME=/workspace/tmp \
		--detach-keys "ctrl-^,ctrl-^"  \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace \
		--workdir /workspace \
		$(GPU_ARGS) \
		$(PAPER_TITLE) \
		python main.py $(ARGS)
	touch tmp/completed

clean:
	rm -rf tmp
