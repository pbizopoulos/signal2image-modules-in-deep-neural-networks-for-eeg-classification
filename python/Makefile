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
	printf 'FROM python\nWORKDIR /usr/src/app\nCOPY pyproject.toml .\nRUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]\n' > $@

bin:
	mkdir $@

bin/all: .dockerignore .gitignore Dockerfile bin main.py pyproject.toml
	docker container run \
		$$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all') \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env HOME=/usr/src/app/bin \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/check: .dockerignore .gitignore Dockerfile bin bin/ruff.toml main.py pyproject.toml
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--env HOME=/usr/src/app/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'mypy --cache-dir bin/ --ignore-missing-imports --install-types --non-interactive --strict main.py && ruff --config bin/ruff.toml main.py'
	touch $@

bin/ruff.toml:
	printf 'select = ["A", "ANN", "ARG", "B", "BLE", "C", "C4", "COM", "E", "EM", "ERA", "EXE", "F", "FBT", "G", "I", "ICN", "INP", "ISC", "N", "NPY", "PD", "PGH", "PIE", "PL", "PTH", "PYI", "Q", "RET", "RSE", "RUF", "S", "SLF", "SIM", "T10", "T20", "TCH", "TID", "TRY", "UP", "W"]\nignore = ["B905", "E501", "PLR0912", "PLR0913", "PLR0915", "PLR2004", "S101"]\nfix = true\ncache-dir = "bin/ruff"\n\n[flake8-quotes]\ninline-quotes = "single"\n' > $@

main.py:
	printf '' > $@

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\ndependencies = []\n\n[project.optional-dependencies]\ndev = [\n\t"mypy",\n\t"ruff",\n]\n' > $@
