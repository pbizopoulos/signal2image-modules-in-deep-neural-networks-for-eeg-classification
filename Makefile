.POSIX:

.PHONY: all check clean help

container_engine=docker # For podman first execute $(printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf)
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir=/work

all: bin/ms.pdf ## Build binaries.

check: bin/check ## Check code.

clean: ## Remove binaries.
	rm -rf bin/

help: ## Show help.
	@sed -n '/sed/d; /##/p' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.* ## "}; {printf "make %-10s# %s\n", $$1, $$2}'

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/check: bin ms.bib ms.tex .dockerignore .gitignore
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		chktex ms.tex && \
		lacheck ms.tex'
	touch bin/check

bin/ms.pdf: bin ms.bib ms.tex .dockerignore .gitignore
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=bin/ ms.tex

ms.bib:
	touch ms.bib

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > ms.tex
