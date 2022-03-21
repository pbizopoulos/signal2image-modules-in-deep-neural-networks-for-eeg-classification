.POSIX:

container_engine=docker
# For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
artifactsdir=artifacts
workdir=/app

user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '--user `id -u`:`id -g`')

# Basic commands.

.PHONY: clean help

codefile=main.py

debug_args=$(shell [ -t 0 ] && printf '%s' '--interactive --tty')
gpus_arg=$(shell [ $(container_engine) = 'docker' ] && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')

$(artifactsdir)/code-run: $(codefile) .dockerignore .gitignore Dockerfile requirements.txt
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--env FULL=$(FULL) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` python3 $(codefile)
	touch $(artifactsdir)/code-run

clean:
	rm -rf $(artifactsdir)/

help:
	@printf 'Basic/Advanced commands:                                                 \n'
	@printf '                                 # Basic commands.                       \n'
	@printf 'make                             # Generate draft artifacts (fast).      \n'
	@printf 'make FULL=1                      # Generate full artifacts (slow).       \n'
	@printf 'make clean                       # Remove artifacts/ directory.          \n'
	@printf 'make help                        # Show basic/advanced commands.         \n'
	@printf '                                 # Advanced commands.                    \n'
	@printf '                                 # code commands.                        \n'
	@printf 'make artifacts/code-coverage     # Code coverage for $(codefile).        \n'
	@printf 'make artifacts/code-format       # Format $(codefile).                   \n'
	@printf 'make artifacts/code-requirements # Generate $(codefile) requirements.txt.\n'
	@printf '                                 # texlive commands.                     \n'
	@printf 'make artifacts/ms.pdf            # Generate document.                    \n'
	@printf 'make artifacts/texlive-lint      # Lint $(texfile).                      \n'
	@printf 'make artifacts/texlive-test      # Test document reproducibility.        \n'
	@printf 'make artifacts/texlive-update    # Update texlive container image.       \n'
	@printf '                                 # arxiv commands.                       \n'
	@printf 'make artifacts/arxiv-ms.pdf      # Generate document from arxiv.         \n'
	@printf 'make artifacts/arxiv.tar         # Generate tar for arxiv.               \n'

$(codefile):
	printf "import os\n\nartifactsdir = os.getenv('ARTIFACTSDIR')\nfull = os.getenv('FULL')\n\n\ndef main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n" > $(codefile)

.dockerignore:
	printf '**\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

requirements.txt:
	printf '# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\npipreqs\npython-minimizer\npyupgrade\n\n# $(codefile) requirements\n' > requirements.txt

# code commands.

.PHONY: $(artifactsdir)/code-requirements

$(artifactsdir)/code-coverage: $(codefile) Dockerfile requirements.txt
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'coverage run $(codefile) && coverage html && rm -rf $(artifactsdir)/htmlcov && mv htmlcov/ $(artifactsdir)/ && mv .coverage $(artifactsdir)/'
	touch $(artifactsdir)/code-coverage

$(artifactsdir)/code-format: $(codefile)
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'python-minimizer --keep-whitespace --out-path $(codefile) $(codefile) && isort $(codefile) && autoflake --expand-star-imports --in-place --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables $(codefile) && pyupgrade $(codefile) && autopep8 -a -a -a --in-place --max-line-length 10000 $(codefile)'
	touch $(artifactsdir)/code-format

$(artifactsdir)/code-requirements:
	mkdir -p $(artifactsdir)/
	rm requirements.txt
	make requirements.txt
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c 'pipreqs --print . >> requirements.txt'

# texlive commands.

.PHONY: $(artifactsdir)/texlive-test $(artifactsdir)/texlive-update

bibfile=ms.bib
texfile=ms.tex

$(artifactsdir)/ms.pdf: $(bibfile) $(texfile) $(artifactsdir)/code-run
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex='\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}' -outdir=$(artifactsdir)/ $(texfile)

$(artifactsdir)/texlive-lint: $(bibfile) $(texfile) $(artifactsdir)/code-run
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'chktex $(texfile) && lacheck $(texfile)'
	touch $(artifactsdir)/texlive-lint

$(artifactsdir)/texlive-test:
	make clean
	make $(artifactsdir)/ms.pdf && cp $(artifactsdir)/ms.pdf $(artifactsdir)/ms-first-run.pdf && touch $(codefile)
	make $(artifactsdir)/ms.pdf && cmp $(artifactsdir)/ms.pdf $(artifactsdir)/ms-first-run.pdf

$(artifactsdir)/texlive-update:
	$(container_engine) image pull texlive/texlive

$(bibfile):
	touch $(bibfile)

$(texfile):
	printf '\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n' > $(texfile)

# arxiv commands.

$(artifactsdir)/arxiv-ms.pdf:
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'wget -U Mozilla -O $(artifactsdir)/arxiv-download.tar https://arxiv.org/e-print/`grep arxiv.org README | cut -d "/" -f5` && tar xfz $(artifactsdir)/arxiv-download.tar'
	rm $(artifactsdir)/arxiv-download.tar
	mv ms.bbl $(artifactsdir)/
	touch $(artifactsdir)/code-run
	make $(artifactsdir)/ms.pdf
	mv $(artifactsdir)/ms.pdf $(artifactsdir)/arxiv-ms.pdf

$(artifactsdir)/arxiv-upload.tar:
	cp $(artifactsdir)/ms.bbl .
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive bash -c 'tar cf $(artifactsdir)/arxiv-upload.tar ms.bbl $(bibfile) $(texfile) `grep "./$(artifactsdir)" $(artifactsdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl
