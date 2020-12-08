.POSIX:

document_title=signal2image-modules-in-deep-neural-networks-for-eeg-classification

ifeq (1, $(shell [ -t 0 ] && echo 1))
	debug_args=--interactive --tty
endif

ifneq (, $(shell which nvidia-smi))
	gpu_args=--gpus all
endif

tmp/ms.pdf: ms.tex ms.bib tmp/results_computed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=tmp/ -cd /workspace/ms.tex

tmp/results_computed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	mkdir -p tmp/
	docker image build --tag $(document_title) .
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--env HOME=/workspace/tmp \
		--detach-keys "ctrl-^,ctrl-^"  \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		--workdir /workspace/ \
		$(document_title) \
		python main.py $(ARG)
	touch tmp/results_computed

clean:
	rm -rf tmp/
