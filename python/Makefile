.POSIX:

.PHONY: all check clean

DEBUG = 1

gpus_all_arg = $$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
interactive_tty_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
python_file_name = main.py

all: bin/all

check: bin/check

clean:
	rm -rf bin/

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
		$(gpus_all_arg) \
		$(interactive_tty_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env HOME=/work/bin \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) python3 $(python_file_name)
	touch bin/all

bin/check: $(python_file_name) bin
	docker container run \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		python /bin/sh -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir https://github.com/pbizopoulos/source-code-simplifier/archive/main.zip && \
		bin/.local/bin/source_code_simplifier $(python_file_name)'
	touch bin/check

Dockerfile:
	printf 'FROM python\nWORKDIR /work\nCOPY pyproject.toml .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir .\n' > Dockerfile

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\ndependencies = []\n' > pyproject.toml
