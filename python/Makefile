.POSIX:

.PHONY: all check clean

DEBUG = 1

gpus_all_docker_arg = $$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')
interactive_tty_docker_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
make_all_docker_cmd = python3 $(python_file_name)
python_file_name = main.py

all: bin/all

check: bin/check

clean:
	rm -rf bin/

$(python_file_name):
	printf '' > $(python_file_name)

.dockerignore:
	printf '*\n!pyproject.toml\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

Dockerfile:
	printf 'FROM python\nENV PIP_NO_CACHE_DIR=1\nWORKDIR /work\nCOPY pyproject.toml .\nRUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]\n' > Dockerfile

bin:
	mkdir bin

bin/all: $(python_file_name) .dockerignore .gitignore Dockerfile bin pyproject.toml
	docker container run \
		$(gpus_all_docker_arg) \
		$(interactive_tty_docker_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env HOME=/work/bin \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch bin/all

bin/check: $(python_file_name) .dockerignore .gitignore Dockerfile bin pyproject.toml
	docker container run \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) ruff --no-cache $(python_file_name)
	touch bin/check

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\ndependencies = []\n\n[project.optional-dependencies]\ndev = ["ruff"]\n\n[tool.ruff]\nselect = ["A", "B", "BLE", "C", "C4", "E", "EM", "ERA", "F", "I", "ICN", "ISC", "N", "PD", "PGH", "PIE", "PLC", "PLE", "PLR", "PLW", "Q", "RET", "RUF", "S", "SIM", "T10", "T20", "TID", "UP", "W"]\nignore = ["B905", "C901", "E501", "PD013", "S101"]\nfix = true\n\n[tool.ruff.flake8-quotes]\ninline-quotes = "single"\n' > pyproject.toml
