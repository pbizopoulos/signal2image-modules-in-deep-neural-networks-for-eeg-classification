.POSIX:

.PHONY: clean help

artifactsdir=artifacts
codefile=main.py
container_engine=docker # For podman first execute $(printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf)
debug_args=$$(test -t 0 && printf '%s' '--interactive --tty')
gpus_arg=$$(test $(container_engine) = 'docker' && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
workdir=/work

$(artifactsdir)/code-run: $(codefile) .dockerignore .gitignore Dockerfile requirements.txt ## Generate draft artifacts (FULL=1 for full).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env FULL=$(FULL) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--volume $$(pwd):$(workdir)/ \
		--workdir $(workdir)/ \
		$$($(container_engine) image build --quiet .) python3 $(codefile)
	touch $(artifactsdir)/code-run

$(artifactsdir)/code-format: $(codefile) ## Format $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--volume $$(pwd):$(workdir)/ \
		--workdir $(workdir)/ \
		python /bin/bash -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source_normalizer/archive/main.zip && \
		$(artifactsdir)/.local/bin/source_normalizer $(codefile) > $(artifactsdir)/tmp.py && \
		mv $(artifactsdir)/tmp.py $(codefile)'
	touch $(artifactsdir)/code-format

$(codefile):
	printf "from os import environ\\n\\n\\ndef main():\\n    artifacts_dir = environ['ARTIFACTSDIR']\\n    full = environ['FULL']\\n\\n\\nif __name__ == '__main__':\\n    main()\\n" > $(codefile)

.dockerignore:
	printf '*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

clean: ## Remove $(artifactsdir) directory.
	rm -rf $(artifactsdir)/

help: ## Show all commands.
	@sed 's/\$$(artifactsdir)/$(artifactsdir)/g; s/\$$(codefile)/$(codefile)/g' $(MAKEFILE_LIST) | grep '##' | grep -v grep | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'

requirements.txt:
	touch requirements.txt
