.POSIX:

.PHONY: clean help

artifacts_dir=artifacts
code_file_name=main.py
container_engine=docker # For podman first execute $(printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf)
debug_args=$$(test -t 0 && printf '%s' '--interactive --tty')
gpus_arg=$$(test $(container_engine) = 'docker' && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir=/work

$(artifacts_dir)/code-run: $(artifacts_dir) $(code_file_name) .dockerignore .gitignore Dockerfile requirements.txt ## Generate draft artifacts (FULL=1 for full).
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTS_DIR=$(work_dir)/$(artifacts_dir) \
		--env FULL=$(FULL) \
		--env HOME=$(work_dir)/$(artifacts_dir) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		$$($(container_engine) image build --quiet .) python3 $(code_file_name)
	touch $(artifacts_dir)/code-run

$(artifacts_dir)/code-format: $(artifacts_dir) $(code_file_name) ## Format $(code_file_name).
	$(container_engine) container run \
		$(user_arg) \
		--env HOME=$(work_dir)/$(artifacts_dir) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		python /bin/bash -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source-code-simplifier/archive/main.zip && \
		$(artifacts_dir)/.local/bin/source_code_simplifier $(code_file_name) > $(artifacts_dir)/tmp.py && \
		mv $(artifacts_dir)/tmp.py $(code_file_name)'
	touch $(artifacts_dir)/code-format

clean: ## Remove $(artifacts_dir) directory.
	rm -rf $(artifacts_dir)/

help: ## Show all commands.
	@sed -n '/sed/d; s/\$$(artifacts_dir)/$(artifacts_dir)/g; s/\$$(code_file_name)/$(code_file_name)/g; /##/p' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'

$(artifacts_dir):
	mkdir $(artifacts_dir)

$(code_file_name):
	printf "from os import environ\\n\\n\\ndef main():\\n    artifacts_dir = environ['ARTIFACTS_DIR']\\n    full = environ['FULL']\\n\\n\\nif __name__ == '__main__':\\n    main()\\n" > $(code_file_name)

.dockerignore:
	printf '*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifacts_dir)/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

requirements.txt:
	touch requirements.txt
