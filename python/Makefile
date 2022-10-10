.POSIX:

.PHONY: all check clean help

container_engine = docker # For podman first execute $(printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf)
debug = 1
debug_args = $$(test -t 0 && printf '%s' '--interactive --tty')
gpus_arg = $$(test $(container_engine) = 'docker' && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
user_arg = $$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir = /work

all: bin/all

check: bin/check

clean:
	rm -rf bin/

help:
	@printf 'make all 	# Build binaries (debug=0 for disabling debug).\n'
	@printf 'make check 	# Check code.\n'
	@printf 'make clean 	# Remove binaries.\n'
	@printf 'make help 	# Show help.\n'

.dockerignore:
	printf '*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/all: .dockerignore .gitignore bin Dockerfile main.py requirements.txt
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=$(work_dir)/bin \
		--env debug=$(debug) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		$$($(container_engine) image build --quiet .) python3 main.py
	touch bin/all

bin/check: bin main.py
	$(container_engine) container run \
		$(user_arg) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		python /bin/bash -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source-code-simplifier/archive/main.zip && \
		bin/.local/bin/source_code_simplifier main.py > bin/tmp.py && \
		mv bin/tmp.py main.py'
	touch bin/check

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

main.py:
	printf "from os import environ\\n\\n\\ndef main():\\n    debug = environ['debug']\\n\\n\\nif __name__ == '__main__':\\n    main()\\n" > main.py

requirements.txt:
	touch requirements.txt
