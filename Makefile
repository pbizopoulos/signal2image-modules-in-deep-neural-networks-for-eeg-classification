.POSIX:

container_engine=docker
# For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
artifactsdir=artifacts
workdir=/app

user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '--user `id -u`:`id -g`')

############################### code commands ##############################

.PHONY: $(artifactsdir)/code-requirements

codefile=main.py

debug_args=$(shell [ -t 0 ] && printf '%s' '--interactive --tty')
gpus_arg=$(shell [ $(container_engine) = 'docker' ] && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')

$(artifactsdir)/code-run: $(codefile) .dockerignore .gitignore Dockerfile requirements.txt ## 		Generate draft artifacts (FULL=1 for full).
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

$(artifactsdir)/code-format: $(codefile) ## 	Format $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		python /bin/bash -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source_normalizer/archive/main.zip && \
		$(artifactsdir)/.local/bin/source_normalizer $(codefile) > $(artifactsdir)/tmp.py && \
		mv $(artifactsdir)/tmp.py $(codefile)'
	touch $(artifactsdir)/code-format

$(codefile):
	printf "import os\n\n\ndef main():\n    artifacts_dir = os.getenv('ARTIFACTSDIR')\n    full = os.getenv('FULL')\n\n\nif __name__ == '__main__':\n    main()\n" > $(codefile)

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

requirements.txt:
	touch requirements.txt

############################### document commands ##########################

.PHONY: $(artifactsdir)/tex-test $(artifactsdir)/texlive-update $(artifactsdir)/pandoc-update

bibfile=ms.bib
texfile=ms.tex

$(artifactsdir)/ms.pdf: .dockerignore .gitignore $(bibfile) $(texfile) $(artifactsdir)/code-run ## 		Generate pdf document.
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -usepretex='\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}' -outdir=$(artifactsdir)/ $(texfile)

$(artifactsdir)/ms-server.pdf: ##	Generate pdf document from server (SERVER_URL=https://arxiv.org/e-print/0000.00000 is required).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c '\
		curl -L -o $(artifactsdir)/download.tar $(SERVER_URL) && \
		tar xfz $(artifactsdir)/download.tar'
	rm $(artifactsdir)/download.tar
	mv ms.bbl $(artifactsdir)/
	touch $(artifactsdir)/code-run
	make $(artifactsdir)/ms.pdf
	mv $(artifactsdir)/ms.pdf $(artifactsdir)/ms-server.pdf

$(artifactsdir)/ms.%: $(bibfile) $(texfile) $(artifactsdir)/code-run ## 		Generate document using pandoc (replace % with the output format).
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		pandoc/latex $(texfile) -o $@

$(artifactsdir)/tex-lint: $(bibfile) $(texfile) $(artifactsdir)/code-run ##	 	Lint $(texfile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c '\
		chktex $(texfile) && \
		lacheck $(texfile)'
	touch $(artifactsdir)/tex-lint

$(artifactsdir)/tex-test: ## 		Test document reproducibility.
	make clean
	make $(artifactsdir)/ms.pdf
	cp $(artifactsdir)/ms.pdf $(artifactsdir)/ms-first-run.pdf
	touch $(codefile)
	make $(artifactsdir)/ms.pdf
	cmp $(artifactsdir)/ms.pdf $(artifactsdir)/ms-first-run.pdf

$(artifactsdir)/tex.tar: $(artifactsdir)/ms.pdf ##		Generate tar file that contains the tex code.
	cp $(artifactsdir)/ms.bbl .
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c 'tar cf $(artifactsdir)/tex.tar ms.bbl $(bibfile) $(texfile) `grep "./$(artifactsdir)" $(artifactsdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

$(artifactsdir)/texlive-update: ## 	Update texlive container image.
	$(container_engine) image pull texlive/texlive

$(artifactsdir)/pandoc-update: ## 	Update pandoc container image.
	$(container_engine) image pull pandoc/latex

$(bibfile):
	touch $(bibfile)

$(texfile):
	printf "\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(texfile)

############################### Makefile commands ##############################

.PHONY: clean help

clean: ## 			Remove artifacts/ directory.
	rm -rf $(artifactsdir)/

help: ## 				Show all commands.
	@grep '##' $(MAKEFILE_LIST) | sed 's/\(\:.*\#\#\)/\:\ /' | sed 's/\$$(artifactsdir)/$(artifactsdir)/' | sed 's/\$$(codefile)/$(codefile)/' | sed 's/\$$(texfile)/$(texfile)/' | grep -v grep

.dockerignore:
	printf '**\n**/.*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore
