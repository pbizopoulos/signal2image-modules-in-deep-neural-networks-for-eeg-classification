.POSIX:

DEBUG = 1

make_all_docker_cmd = python3 main.py

all: bin/all

check: bin/check

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n!pyproject.toml\n' > $@

.gitignore:
	printf 'bin/\n' > $@

Dockerfile:
	printf 'FROM python\nWORKDIR /work\nCOPY pyproject.toml .\nRUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]\n' > $@

bin:
	mkdir $@

bin/all: .dockerignore .gitignore Dockerfile bin main.py pyproject.toml
	docker container run \
		$$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all') \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/check: .dockerignore .gitignore Dockerfile bin main.py pyproject.toml
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) ruff main.py
	touch $@

main.py:
	printf '' > $@

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\ndependencies = []\n\n[project.optional-dependencies]\ndev = ["ruff"]\n\n[tool.ruff]\nselect = ["A", "B", "BLE", "C", "C4", "E", "EM", "ERA", "F", "I", "ICN", "ISC", "N", "PD", "PGH", "PIE", "PLC", "PLE", "PLR", "PLW", "Q", "RET", "RUF", "S", "SIM", "T10", "T20", "TID", "UP", "W"]\nignore = ["B905", "C901", "E501", "PLR0913", "PLR2004", "S101"]\nfix = true\ncache-dir = "bin/ruff"\n\n[tool.ruff.flake8-quotes]\ninline-quotes = "single"\n' > $@
