.POSIX:

.PHONY: all clean help

container_engine=docker # For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
debug_args=$$(test -t 0 && printf '%s' '--interactive --tty')
user_arg=$$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir=/work

all: .dockerignore .gitignore bin/cert.pem ## Run server.
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--publish 8080:8080 \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/bash -c 'npx --yes http-server --cert bin/cert.pem --key bin/key.pem --ssl'

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

bin/cert.pem: bin
	openssl req -newkey rsa:2048 -subj "/C=../ST=../L=.../O=.../OU=.../CN=.../emailAddress=..." -new -nodes -x509 -days 3650 -keyout bin/key.pem -out bin/cert.pem

bin/check: .dockerignore .gitignore bin index.html
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/bash -c 'npx --yes html-validate index.html'
	touch bin/check

index.html:
	printf '\n' > index.html
