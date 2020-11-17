.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
PAPER_TITLE=signal2image-modules-in-deep-neural-networks-for-eeg-classification

ifeq (, $(shell which nvidia-smi))
	GPU_ARGS=
else
	GPU_ARGS=--gpus all
endif

cache/ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/usr/src/app \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -outdir=cache/ -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /usr/src/app/ms.tex
	@if [ -f cache/.ms-latest.pdf ]; then \
		cmp cache/ms.pdf cache/.ms-latest.pdf && echo 'cache/ms.pdf unchanged.' || echo 'cache/ms.pdf changed.'; fi
	@cp cache/ms.pdf cache/ms-`date --iso-8601=seconds`.pdf
	@cp cache/ms.pdf cache/.ms-latest.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/*
	docker image build --tag $(PAPER_TITLE) .
	docker container run \
		$(DEBUG_ARGS) \
		--env HOME=/usr/src/app/cache/ \
		--detach-keys "ctrl-^,ctrl-^"  \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/usr/src/app \
		--workdir /usr/src/app \
		$(GPU_ARGS) \
		$(PAPER_TITLE) \
		python main.py cache/ results/ $(ARGS)
	touch results/completed

clean:
	rm -rf __pycache__/ cache/* results/*
