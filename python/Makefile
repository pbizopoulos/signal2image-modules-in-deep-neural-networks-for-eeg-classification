.POSIX:

DEBUG = 1

make_all_docker_cmd = python3 main.py

all: bin/done

check: bin/check/done

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

bin/done: .dockerignore .gitignore Dockerfile bin main.py pyproject.toml
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
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/check/done: .dockerignore .gitignore Dockerfile bin main.py pyproject.toml
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--env HOME=/usr/src/app/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'mypy main.py && ruff main.py'
	touch $@

main.py:
	printf '' > $@

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\ndependencies = []\n\n[project.optional-dependencies]\ndev = [\n\t"mypy",\n\t"ruff",\n]\n\n[tool.mypy]\ncache_dir = "bin/check/mypy"\nignore_missing_imports = true\ninstall_types = true\nnon_interactive = true\nstrict = true\n\n[tool.ruff]\ncache-dir = "bin/check/ruff"\nfix = true\nignore = ["E501", "S101"]\nselect = ["A", "ANN", "ARG", "B", "BLE", "C", "C40", "C90", "COM", "DJ", "DTZ", "E", "EM", "ERA", "EXE", "F", "FBT", "G", "I", "ICN", "INP", "ISC", "N", "NPY", "PD", "PGH", "PIE", "PL", "PT", "PTH", "PYI", "Q", "RET", "RSE", "RUF", "S", "SLF", "SIM", "T10", "T20", "TCH", "TID", "TRY", "UP", "YTT", "W"]\n' > $@
