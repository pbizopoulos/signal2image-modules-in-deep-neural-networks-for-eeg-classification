.POSIX:

.PHONY: $(artifacts_dir)/texlive-update $(artifacts_dir)/pandoc-update clean help

artifacts_dir=artifacts
bib_file_name=ms.bib
code_file_name=ms.tex
container_engine=docker # For podman first execute $(printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf)
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir=/work

$(artifacts_dir)/ms.pdf: $(artifacts_dir) $(bib_file_name) $(code_file_name) .dockerignore .gitignore ## Generate pdf document.
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=$(artifacts_dir)/ $(code_file_name)

$(artifacts_dir)/ms.%: $(artifacts_dir) $(bib_file_name) $(code_file_name) ## Generate document using pandoc (replace % with the output format).
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		pandoc/latex $(code_file_name) -o $@

$(artifacts_dir)/ms-server.pdf: $(artifacts_dir) ## Generate pdf document from server (SERVER_URL=https://arxiv.org/e-print/0000.00000 is required).
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		curl -L -o $(artifacts_dir)/download.tar $(SERVER_URL) && \
		tar xfz $(artifacts_dir)/download.tar'
	rm $(artifacts_dir)/download.tar
	mv ms.bbl $(artifacts_dir)/
	make $(artifacts_dir)/ms.pdf
	mv $(artifacts_dir)/ms.pdf $(artifacts_dir)/ms-server.pdf

$(artifacts_dir)/tex.tar: $(artifacts_dir)/ms.pdf ## Generate tar file that contains the tex code.
	touch $(artifacts_dir)/ms.bbl && cp $(artifacts_dir)/ms.bbl .
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c 'tar cf $(artifacts_dir)/tex.tar ms.bbl $(bib_file_name) $(code_file_name) $$(grep "^INPUT ./" $(artifacts_dir)/ms.fls | uniq | cut -b 9-)'
	rm ms.bbl

$(artifacts_dir)/tex-lint: $(artifacts_dir) $(bib_file_name) $(code_file_name) ## Lint $(code_file_name).
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		chktex $(code_file_name) && \
		lacheck $(code_file_name)'
	touch $(artifacts_dir)/tex-lint

$(artifacts_dir)/texlive-update: ## Update texlive container image.
	$(container_engine) image pull texlive/texlive

$(artifacts_dir)/pandoc-update: ## Update pandoc container image.
	$(container_engine) image pull pandoc/latex

clean: ## Remove $(artifacts_dir) directory.
	rm -rf $(artifacts_dir)/

help: ## Show all commands.
	@sed 's/\$$(artifacts_dir)/$(artifacts_dir)/g; s/\$$(code_file_name)/$(code_file_name)/g' $(MAKEFILE_LIST) | grep '##' | grep -v grep | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'

$(artifacts_dir):
	mkdir -p $(artifacts_dir)

$(bib_file_name):
	touch $(bib_file_name)

$(code_file_name):
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(code_file_name)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf '$(artifacts_dir)/\n' > .gitignore
