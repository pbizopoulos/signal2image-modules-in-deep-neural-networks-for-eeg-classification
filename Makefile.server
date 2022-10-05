.POSIX:

.PHONY: all clean help

artifacts_dir=artifacts
code_file_name=index.html
container_engine=docker # For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
debug_args=$$(test -t 0 && printf '%s' '--interactive --tty')
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir=/work

all: $(artifacts_dir)/cert.pem .gitignore ## Run server.
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(work_dir)/$(artifacts_dir) \
		--env NODE_PATH=$(work_dir)/$(artifacts_dir) \
		--publish 8080:8080 \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/bash -c 'npx --yes http-server -S -C $(artifacts_dir)/cert.pem -K $(artifacts_dir)/key.pem'

$(artifacts_dir)/code-analyze: $(artifacts_dir) $(code_file_name) .gitignore ## Analyze code.
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(work_dir)/$(artifacts_dir) \
		--env NODE_PATH=$(work_dir)/$(artifacts_dir) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/bash -c 'npx --yes html-validate $(code_file_name)'
	touch $(artifacts_dir)/code-analyze

clean: ## Remove dependent directories.
	rm -rf $(artifacts_dir)/

help: ## Show all commands.
	@sed -n '/sed/d; s/\$$(artifacts_dir)/$(artifacts_dir)/g; /##/p' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'

$(artifacts_dir):
	mkdir $(artifacts_dir)

$(artifacts_dir)/cert.pem: $(artifacts_dir)
	openssl req -newkey rsa:2048 -subj "/C=../ST=../L=.../O=.../OU=.../CN=.../emailAddress=..." -new -nodes -x509 -days 3650 -keyout $(artifacts_dir)/key.pem -out $(artifacts_dir)/cert.pem

.gitignore:
	printf '$(artifacts_dir)/\n' > .gitignore
