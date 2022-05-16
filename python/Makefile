.POSIX:

.PHONY: $(artifactsdir)/code-requirements clean help

artifactsdir=artifacts
codefile=main.py
container_engine=docker # For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
debug_args=$(shell [ -t 0 ] && printf '%s' '--interactive --tty')
gpus_arg=$(shell [ $(container_engine) = 'docker' ] && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '`id -u`:`id -g`')
workdir=/app

$(artifactsdir)/code-run: $(codefile) .dockerignore .gitignore Dockerfile requirements.txt ## 	Generate draft artifacts (FULL=1 for full).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env FULL=$(FULL) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` python3 $(codefile)
	touch $(artifactsdir)/code-run

$(artifactsdir)/code-format: $(codefile) ## 	Format $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--user $(user_arg) \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		python /bin/bash -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source_normalizer/archive/main.zip && \
		$(artifactsdir)/.local/bin/source_normalizer $(codefile) > $(artifactsdir)/tmp.py && \
		mv $(artifactsdir)/tmp.py $(codefile)'
	touch $(artifactsdir)/code-format

$(codefile):
	printf "from os import environ\n\n\ndef main():\n    artifacts_dir = environ['ARTIFACTSDIR']\n    full = environ['FULL']\n\n\nif __name__ == '__main__':\n    main()\n" > $(codefile)

.dockerignore:
	printf '*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

clean: ## 			Remove artifacts/ directory.
	rm -rf $(artifactsdir)/

help: ## 			Show all commands.
	@grep '##' $(MAKEFILE_LIST) | sed 's/\(\:.*\#\#\)//' | sed 's/\$$(artifactsdir)/$(artifactsdir)/' | sed 's/\$$(codefile)/$(codefile)/' | grep -v grep

requirements.txt:
	touch requirements.txt
