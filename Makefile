.POSIX:

is_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_interactive
	debug_args=--interactive --tty
endif

ifneq (, $(shell which nvidia-container-cli))
	gpu_args=--gpus all
endif

tmp/ms.pdf: ms.bib ms.tex tmp/results_computed
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=tmp/ -cd /workspace/ms.tex

tmp/results_computed: Dockerfile main.py requirements.txt
	mkdir -p tmp/
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--env HOME=/workspace/tmp \
		--detach-keys "ctrl-^,ctrl-^"  \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		--workdir /workspace/ \
		`docker image build -q .` \
		python main.py $(VER)
	touch tmp/results_computed

clean:
	rm -rf tmp/
