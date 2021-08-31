.POSIX:

is_shell_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_shell_interactive
	debug_args=--interactive --tty
endif

ifneq (, $(shell command -v nvidia-container-cli))
	gpu_args=--gpus all
endif

container_volume=/workspace
tmp_directory=tmp

$(tmp_directory)/ms.pdf: ms.bib ms.tex $(tmp_directory)/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=$(tmp_directory)/ ms.tex
	if [ ! -z $(VERSION) ] ; then rm -f tmp/for-upload-to-arxiv.tar && make tmp/for-upload-to-arxiv.tar ; fi

$(tmp_directory)/execute-python: Dockerfile main.py requirements.txt
	mkdir -p $(tmp_directory)/
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(container_volume)/$(tmp_directory)/ \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		`docker image build -q .` python main.py $(VERSION)
	touch $(tmp_directory)/execute-python

clean:
	rm -rf $(tmp_directory)/

$(tmp_directory)/format-python: main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		alphachai/isort main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		peterevans/autopep8 -i --max-line-length 1000 main.py
	touch $(tmp_directory)/format-python

$(tmp_directory)/lint-texlive: ms.bib ms.tex $(tmp_directory)/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive bash -c 'chktex ms.tex && lacheck ms.tex'
	touch $(tmp_directory)/lint-texlive

$(tmp_directory)/for-upload-to-arxiv.tar:
	cp $(tmp_directory)/ms.bbl .
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive bash -c 'tar cf $(tmp_directory)/for-upload-to-arxiv.tar ms.bbl ms.bib ms.tex `grep "./$(tmp_directory)" $(tmp_directory)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

$(tmp_directory)/ms-from-arxiv.pdf:
	mkdir -p $(tmp_directory)/
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		curlimages/curl -o $(tmp_directory)/download-from-arxiv.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d '/' -f5`
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive tar xfz $(tmp_directory)/download-from-arxiv.tar
	rm $(tmp_directory)/download-from-arxiv.tar
	mv ms.bbl $(tmp_directory)/
	touch $(tmp_directory)/execute-python
	make $(tmp_directory)/ms.pdf
	mv $(tmp_directory)/ms.pdf $(tmp_directory)/ms-from-arxiv.pdf

$(tmp_directory)/update-makefile:
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		curlimages/curl -LO https://github.com/pbizopoulos/a-makefile-for-developing-containerized-latex-technical-documents/raw/master/Makefile

$(tmp_directory)/update-docker-images:
	docker image pull alphachai/isort
	docker image pull curlimages/curl
	docker image pull peterevans/autopep8
	docker image pull texlive/texlive
