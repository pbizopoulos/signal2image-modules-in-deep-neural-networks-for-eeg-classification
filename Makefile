.POSIX:

.PHONY: texlive-update clean check

is_shell_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_shell_interactive
	debug_args=--interactive --tty
endif

ifneq (, $(shell which nvidia-container-cli))
	gpu_args=--gpus all
endif

bibfile=ms.bib
pythonfile=main.py
texfile=ms.tex
tmpdir=tmp
workdir=/app

$(tmpdir)/python-run: .dockerignore .gitignore Dockerfile $(pythonfile) requirements.txt
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
		`docker image build -q .` python3 $(pythonfile)
	touch $(tmpdir)/python-run

$(tmpdir)/python-coverage: Dockerfile $(pythonfile) requirements.txt
	mkdir -p $(tmpdir)/
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` bash -c "coverage run $(pythonfile) && coverage html && rm -rf $(tmpdir)/htmlcov && mv htmlcov/ $(tmpdir)/ && mv .coverage $(tmpdir)/"
	touch $(tmpdir)/python-coverage

$(tmpdir)/python-format: Dockerfile $(pythonfile)
	mkdir -p $(tmpdir)/
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` bash -c "isort $(pythonfile) && autoflake --in-place --remove-all-unused-imports --remove-unused-variables $(pythonfile) && autopep8 -i --max-line-length 1000 $(pythonfile)"
	touch $(tmpdir)/python-format

$(tmpdir)/ms.pdf: $(bibfile) $(texfile) $(tmpdir)/python-run
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=$(tmpdir)/ $(texfile)

$(tmpdir)/texlive-lint: $(bibfile) $(texfile) $(tmpdir)/python-run
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'chktex $(texfile) && lacheck $(texfile)'
	touch $(tmpdir)/texlive-lint

$(tmpdir)/texlive-update:
	docker image pull texlive/texlive

$(tmpdir)/arxiv-upload.tar:
	cp $(tmpdir)/ms.bbl .
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'tar cf $(tmpdir)/arxiv-upload.tar ms.bbl $(bibfile) $(texfile) `grep "./$(tmpdir)" $(tmpdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

$(tmpdir)/arxiv-download.pdf:
	mkdir -p $(tmpdir)/
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive wget -U Mozilla -O $(tmpdir)/arxiv-download.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d '/' -f5`
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive tar xfz $(tmpdir)/arxiv-download.tar
	rm $(tmpdir)/arxiv-download.tar
	mv ms.bbl $(tmpdir)/
	touch $(tmpdir)/python-run
	make $(tmpdir)/ms.pdf
	mv $(tmpdir)/ms.pdf $(tmpdir)/arxiv-download.pdf

$(pythonfile):
	printf "import os\n\ntmpdir = os.getenv('TMPDIR')\nfull = os.getenv('FULL')\n\n\ndef main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n" > $(pythonfile)

Dockerfile:
	printf "FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n" > Dockerfile

requirements.txt:
	printf "# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\n\n# document requirements\n\n" > requirements.txt

$(texfile):
	printf "\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(texfile)

$(bibfile):
	touch $(bibfile)

.gitignore:
	printf "tmp/\n" > .gitignore

.dockerignore:
	printf ".git/\ntmp/\n" > .dockerignore

clean:
	rm -rf $(tmpdir)/

check:
	make clean
	make $(tmpdir)/ms.pdf && cp $(tmpdir)/ms.pdf $(tmpdir)/ms-previous.pdf && touch $(pythonfile)
	make $(tmpdir)/ms.pdf && cmp $(tmpdir)/ms.pdf $(tmpdir)/ms-previous.pdf
