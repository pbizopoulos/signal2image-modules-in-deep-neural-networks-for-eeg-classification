.POSIX:

container_engine=docker
# For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
tmpdir=tmp
workdir=/app

user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '--user `id -u`:`id -g`')

# Basic commands.

.PHONY: clean help

pythonfile=main.py

debug_args=$(shell [ -t 0 ] && printf '%s' '--interactive --tty')
gpus_arg=$(shell [ $(container_engine) = 'docker' ] && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')

$(tmpdir)/python-run: $(pythonfile) .dockerignore .gitignore Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(workdir)/$(tmpdir) \
		--env FULL=$(FULL) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` python3 $(pythonfile)
	touch $(tmpdir)/python-run

clean:
	rm -rf $(tmpdir)/

help:
	@printf 'Basic/Advanced commands:                                               \n'
	@printf '                             # Basic commands.                         \n'
	@printf 'make                         # Generate draft (fast) results.          \n'
	@printf 'make FULL=1                  # Generate full (slow) results.           \n'
	@printf 'make clean                   # Remove tmp/ directory.                  \n'
	@printf 'make help                    # Show basic/advanced commands.           \n'
	@printf '                             # Advanced commands.                      \n'
	@printf '                             # python commands.                        \n'
	@printf 'make tmp/python-coverage     # Code coverage for $(pythonfile).        \n'
	@printf 'make tmp/python-format       # Format $(pythonfile).                   \n'
	@printf 'make tmp/python-requirements # Generate $(pythonfile) requirements.txt.\n'
	@printf '                             # texlive commands.                       \n'
	@printf 'make tmp/ms.pdf              # Generate document.                      \n'
	@printf 'make tmp/texlive-lint        # Lint $(texfile).                        \n'
	@printf 'make tmp/texlive-test        # Test document reproducibility.          \n'
	@printf 'make tmp/texlive-update      # Update texlive container image.         \n'
	@printf '                             # arxiv commands.                         \n'
	@printf 'make tmp/arxiv-ms.pdf        # Generate document from arxiv.           \n'
	@printf 'make tmp/arxiv.tar           # Generate tar for arxiv.                 \n'

$(pythonfile):
	printf "import os\n\ntmpdir = os.getenv('TMPDIR')\nfull = os.getenv('FULL')\n\n\ndef main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n" > $(pythonfile)

.dockerignore:
	printf '.git/\ntmp/\n' > .dockerignore

.gitignore:
	printf 'tmp/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

requirements.txt:
	printf '# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\npipreqs\npython-minimizer\npyupgrade\n\n# $(pythonfile) requirements\n' > requirements.txt

# python commands.

.PHONY: $(tmpdir)/python-requirements

$(tmpdir)/python-coverage: $(pythonfile) Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'coverage run $(pythonfile) && coverage html && rm -rf $(tmpdir)/htmlcov && mv htmlcov/ $(tmpdir)/ && mv .coverage $(tmpdir)/'
	touch $(tmpdir)/python-coverage

$(tmpdir)/python-format: $(pythonfile)
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'python-minimizer --keep-whitespace --out-path $(pythonfile) $(pythonfile) && isort $(pythonfile) && autoflake --expand-star-imports --in-place --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables $(pythonfile) && pyupgrade $(pythonfile) && autopep8 -a -a -a --in-place --max-line-length 10000 $(pythonfile)'
	touch $(tmpdir)/python-format

$(tmpdir)/python-requirements:
	mkdir -p $(tmpdir)/
	rm requirements.txt
	make requirements.txt
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'pipreqs --print . >> requirements.txt'

# texlive commands.

.PHONY: $(tmpdir)/texlive-test $(tmpdir)/texlive-update

bibfile=ms.bib
texfile=ms.tex

$(tmpdir)/ms.pdf: $(bibfile) $(texfile) $(tmpdir)/python-run
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex='\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}' -outdir=$(tmpdir)/ $(texfile)

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
	printf '\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n' > $(texfile)

# arxiv commands.

$(tmpdir)/arxiv-ms.pdf:
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'wget -U Mozilla -O $(tmpdir)/arxiv-download.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d "/" -f5` && tar xfz $(tmpdir)/arxiv-download.tar'
	rm $(tmpdir)/arxiv-download.tar
	mv ms.bbl $(tmpdir)/
	touch $(tmpdir)/python-run
	make $(tmpdir)/ms.pdf
	mv $(tmpdir)/ms.pdf $(tmpdir)/arxiv-ms.pdf

$(tmpdir)/arxiv-upload.tar:
	cp $(tmpdir)/ms.bbl .
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'tar cf $(tmpdir)/arxiv-upload.tar ms.bbl $(bibfile) $(texfile) `grep "./$(tmpdir)" $(tmpdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl
