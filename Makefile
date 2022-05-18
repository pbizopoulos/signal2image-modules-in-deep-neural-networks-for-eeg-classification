.POSIX:

.PHONY: $(artifactsdir)/tex-test $(artifactsdir)/texlive-update $(artifactsdir)/pandoc-update clean help

artifactsdir=artifacts
bibfile=ms.bib
codefile=ms.tex
container_engine=docker # For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '`id -u`:`id -g`')
workdir=/app

$(artifactsdir)/ms.pdf: $(bibfile) $(codefile) .dockerignore .gitignore ## Generate pdf document.
	$(container_engine) container run \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=$(artifactsdir)/ $(codefile)

$(artifactsdir)/ms.%: $(bibfile) $(codefile) ## Generate document using pandoc (replace % with the output format).
	$(container_engine) container run \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		pandoc/latex $(codefile) -o $@

$(artifactsdir)/ms-server.pdf: ## Generate pdf document from server (SERVER_URL=https://arxiv.org/e-print/0000.00000 is required).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c '\
		curl -L -o $(artifactsdir)/download.tar $(SERVER_URL) && \
		tar xfz $(artifactsdir)/download.tar'
	rm $(artifactsdir)/download.tar
	mv ms.bbl $(artifactsdir)/
	make $(artifactsdir)/ms.pdf
	mv $(artifactsdir)/ms.pdf $(artifactsdir)/ms-server.pdf

$(artifactsdir)/tex.tar: $(artifactsdir)/ms.pdf ## Generate tar file that contains the tex code.
	cp $(artifactsdir)/ms.bbl .
	$(container_engine) container run \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c 'tar cf $(artifactsdir)/tex.tar ms.bbl $(bibfile) $(codefile) `grep "^INPUT ./" $(artifactsdir)/ms.fls | uniq | cut -b 9-`'
	rm ms.bbl

$(artifactsdir)/tex-lint: $(bibfile) $(codefile) ## Lint $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		texlive/texlive /bin/bash -c '\
		chktex $(codefile) && \
		lacheck $(codefile)'
	touch $(artifactsdir)/tex-lint

$(artifactsdir)/texlive-update: ## Update texlive container image.
	$(container_engine) image pull texlive/texlive

$(artifactsdir)/pandoc-update: ## Update pandoc container image.
	$(container_engine) image pull pandoc/latex

$(bibfile):
	touch $(bibfile)

$(codefile):
	printf "\documentclass{article}\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(codefile)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore

clean: ## Remove $(artifactsdir) directory.
	rm -rf $(artifactsdir)/

help: ## Show all commands.
	@sed 's/\$$(artifactsdir)/$(artifactsdir)/g; s/\$$(codefile)/$(codefile)/g' $(MAKEFILE_LIST) | grep '##' | grep -v grep | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'
