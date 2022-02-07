.POSIX:

container_engine=docker
# For podman first execute `echo 'unqualified-search-registries=["docker.io"]' > /etc/containers/registries.conf.d/docker.conf`
tmpdir=tmp
workdir=/app

# For python

.PHONY: clean

debug_args_0=
debug_args_1=--interactive --tty
debug_args=$(debug_args_$(shell [ -t 0 ] && echo 1))

gpus_arg_0_docker=
gpus_arg_1_docker=--gpus all
gpus_arg_0_podman=
gpus_arg_1_podman=
gpus_arg=$(gpus_arg_$(shell which nvidia-container-toolkit > /dev/null && echo 1)_$(container_engine))

user_arg_podman=
user_arg_docker=--user `id -u`:`id -g`
user_arg=$(user_arg_$(container_engine))

pythonfile=main.py

$(tmpdir)/python-run: $(pythonfile) .dockerignore .gitignore Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env FULL=$(FULL) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build -q .` python3 $(pythonfile)
	touch $(tmpdir)/python-run

$(tmpdir)/python-coverage: $(pythonfile) Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build -q .` bash -c "coverage run $(pythonfile) && coverage html && rm -rf $(tmpdir)/htmlcov && mv htmlcov/ $(tmpdir)/ && mv .coverage $(tmpdir)/"
	touch $(tmpdir)/python-coverage

$(tmpdir)/python-format: $(pythonfile)
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build -q .` bash -c "isort $(pythonfile) && autoflake --in-place --remove-all-unused-imports --remove-unused-variables $(pythonfile) && autopep8 -i --max-line-length 1000 $(pythonfile)"
	touch $(tmpdir)/python-format

clean:
	rm -rf $(tmpdir)/

$(pythonfile):
	printf "import os\n\ntmpdir = os.getenv('TMPDIR')\nfull = os.getenv('FULL')\n\n\ndef main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n" > $(pythonfile)

.dockerignore:
	printf ".git/\ntmp/\n" > .dockerignore

.gitignore:
	printf "tmp/\n" > .gitignore

Dockerfile:
	printf "FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n" > Dockerfile

requirements.txt:
	printf "# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\n\n# document requirements\n\n" > requirements.txt

# [Optional] For texlive

.PHONY: $(tmpdir)/texlive-test $(tmpdir)/texlive-update

bibfile=ms.bib
texfile=ms.tex

$(tmpdir)/ms.pdf: $(bibfile) $(texfile) $(tmpdir)/python-run
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=$(tmpdir)/ $(texfile)

$(tmpdir)/texlive-lint: $(bibfile) $(texfile) $(tmpdir)/python-run
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'chktex $(texfile) && lacheck $(texfile)'
	touch $(tmpdir)/texlive-lint

$(tmpdir)/texlive-test:
	make clean
	make $(tmpdir)/ms.pdf && cp $(tmpdir)/ms.pdf $(tmpdir)/ms-first-run.pdf && touch $(pythonfile)
	make $(tmpdir)/ms.pdf && cmp $(tmpdir)/ms.pdf $(tmpdir)/ms-first-run.pdf

$(tmpdir)/texlive-update:
	$(container_engine) image pull texlive/texlive

$(bibfile):
	touch $(bibfile)

$(texfile):
	printf "\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(texfile)

# [Optional] For arxiv

$(tmpdir)/arxiv-download.pdf:
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive wget -U Mozilla -O $(tmpdir)/arxiv-download.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d '/' -f5`
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive tar xfz $(tmpdir)/arxiv-download.tar
	rm $(tmpdir)/arxiv-download.tar
	mv ms.bbl $(tmpdir)/
	touch $(tmpdir)/python-run
	make $(tmpdir)/ms.pdf
	mv $(tmpdir)/ms.pdf $(tmpdir)/arxiv-download.pdf

$(tmpdir)/arxiv-upload.tar:
	cp $(tmpdir)/ms.bbl .
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'tar cf $(tmpdir)/arxiv-upload.tar ms.bbl $(bibfile) $(texfile) `grep "./$(tmpdir)" $(tmpdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

# For app

appfile=app.py

index.html: $(appfile) Dockerfile.app app-requirements.txt
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build -q -f Dockerfile.app .` python3 $(appfile)

serve: index.html
	$(container_engine) container run \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--publish 8000:8000 \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		python python3 -m http.server --directory .

$(appfile):
	printf "from pyclientsideml import generate_page\n\n\n\ndef main():\n    generate_page('signal-classification', 'tmp/')\n\n\nif __name__ == '__main__':\n    main()\n" > $(appfile)

Dockerfile.app:
	printf "FROM python\nCOPY app-requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip wheel && python3 -m pip install --no-cache-dir -r app-requirements.txt\n" > Dockerfile.app

app-requirements.txt:
	printf "https://github.com/pbizopoulos/pyclientsideml/tarball/master\n" > app-requirements.txt

help:
	@echo "Advanced usage:							"
	@echo "make				# Generate draft (fast) results."
	@echo "make FULL=1			# Generate full (slow) results.	"
	@echo "make tmp/python-coverage 	# Code coverage for main.py.	"
	@echo "make tmp/python-format 		# Format main.py.		"
	@echo "make clean 			# Remove tmp/ directory.	"
	@echo "make tmp/ms.pdf			# Generate document.		"
	@echo "make tmp/texlive-lint 		# Lint ms.tex.			"
	@echo "make tmp/texlive-test 		# Test document reproducibility."
	@echo "make tmp/texlive-update		# Update texlive container image.	"
	@echo "make tmp/arxiv-download.pdf	# Generate document from arxiv.	"
	@echo "make tmp/arxiv-upload.tar	# Generate tar for arxiv.	"
	@echo "make index.html			# Generate index.html for app.	"
	@echo "make serve			# Serve index.html for app.	"
	@echo "make help			# Show help.			"
