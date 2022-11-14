.POSIX:

.PHONY: all check clean help

DEBUG = 1

debug_args = $$(test -t 0 && printf '%s' '--interactive --tty')
gpus_arg = $$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
python_file_name = main.py
work_dir = /work

all: bin/all

check: bin/check

clean:
	rm -rf bin/

help:
	@printf 'make all	# Build binaries (DEBUG=0 for disabling debug).\n'
	@printf 'make check	# Check code.\n'
	@printf 'make clean	# Remove binaries.\n'
	@printf 'make help	# Show help.\n'

$(python_file_name):
	printf "from os import environ\\n\\n\\ndef main():\\n    debug = environ['DEBUG']\\n\\n\\nif __name__ == '__main__':\\n    main()\\n" > $(python_file_name)

.dockerignore:
	printf '*\n!pyproject.toml\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/all: $(python_file_name) .dockerignore .gitignore bin Dockerfile pyproject.toml
	docker container run \
		$(debug_args) \
		$(gpus_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env HOME=$(work_dir)/bin \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		$$(docker image build --quiet .) python3 $(python_file_name)
	touch bin/all

bin/check: $(python_file_name) bin
	docker container run \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		python /bin/sh -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source-code-simplifier/archive/main.zip && \
		bin/.local/bin/source_code_simplifier $(python_file_name)'
	touch bin/check

Dockerfile:
	printf 'FROM python\nWORKDIR $(work_dir)\nCOPY pyproject.toml .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir .\n' > Dockerfile

pyproject.toml:
	printf '[project]\nname = "None"\nversion = "0"\ndependencies = []\n' > pyproject.toml
