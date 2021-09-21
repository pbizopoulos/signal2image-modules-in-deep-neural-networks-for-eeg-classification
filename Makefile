.POSIX:

is_shell_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_shell_interactive
	debug_args=--interactive --tty
endif

ifneq (, $(shell which nvidia-container-cli))
	gpu_args=--gpus all
endif

tmpdir=tmp
workdir=/app

$(tmpdir)/ms.pdf: ms.bib ms.tex $(tmpdir)/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=$(tmpdir)/ ms.tex
	if [ ! -z $(FULL) ] ; then rm -f $(tmpdir)/for-upload-to-arxiv.tar && make $(tmpdir)/for-upload-to-arxiv.tar ; fi

$(tmpdir)/execute-python: Dockerfile main.py requirements.txt
	mkdir -p $(tmpdir)/
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env FULL=$(FULL) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` python main.py
	touch $(tmpdir)/execute-python

clean:
	rm -rf $(tmpdir)/

$(tmpdir)/coverage-python: Dockerfile main.py requirements.txt
	mkdir -p $(tmpdir)/
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env FULL=$(FULL) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` bash -c "coverage run main.py && coverage html && rm -rf $(tmpdir)/htmlcov && mv htmlcov/ $(tmpdir)/ && mv .coverage $(tmpdir)/"
	touch $(tmpdir)/coverage-python

$(tmpdir)/format-python: main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` bash -c "isort main.py && autopep8 -i --max-line-length 1000 main.py"
	touch $(tmpdir)/format-python

$(tmpdir)/lint-texlive: ms.bib ms.tex $(tmpdir)/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'chktex ms.tex && lacheck ms.tex'
	touch $(tmpdir)/lint-texlive

$(tmpdir)/for-upload-to-arxiv.tar:
	cp $(tmpdir)/ms.bbl .
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'tar cf $(tmpdir)/for-upload-to-arxiv.tar ms.bbl ms.bib ms.tex `grep "./$(tmpdir)" $(tmpdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

$(tmpdir)/ms-from-arxiv.pdf:
	mkdir -p $(tmpdir)/
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive wget -U Mozilla -O $(tmpdir)/download-from-arxiv.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d '/' -f5`
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive tar xfz $(tmpdir)/download-from-arxiv.tar
	rm $(tmpdir)/download-from-arxiv.tar
	mv ms.bbl $(tmpdir)/
	touch $(tmpdir)/execute-python
	make $(tmpdir)/ms.pdf
	mv $(tmpdir)/ms.pdf $(tmpdir)/ms-from-arxiv.pdf

$(tmpdir)/update-texlive:
	docker image pull texlive/texlive
