.POSIX:

DEBUG = 1

cmd = python3 main.py

all: bin/done

check: tmp/done

clean:
	rm -rf bin/ data/ tmp/

.dockerignore:
	printf '*\n!pyproject.toml\n' > $@

.gitignore:
	printf '.env\nbin/\ndata/\ntmp/\n' > $@

Dockerfile:
	printf 'FROM python\nWORKDIR /usr/src/app\nCOPY pyproject.toml .\nRUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]\n' > $@

bin/done: .dockerignore .gitignore Dockerfile tmp main.py pyproject.toml
	docker container run \
		$$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all') \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--env HOME=/usr/src/app/tmp \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(cmd)
	touch $@

main.py:
	touch $@

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\n\n[project.optional-dependencies]\ndev = [\n\t"mypy",\n\t"ruff",\n]\n' > $@

tmp:
	mkdir $@

tmp/done: .dockerignore .gitignore Dockerfile tmp main.py pyproject.toml
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--env HOME=/usr/src/app/tmp \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'mypy --cache-dir tmp/mypy --ignore-missing-imports --install-types --non-interactive --strict main.py && ruff --cache-dir tmp/ruff --fix --ignore D,E501,S101 --select ALL main.py'
	if ls -a | grep -v -E -x '.|..|.dockerignore|.env|.gitignore|__init__.py|Dockerfile|Makefile|assets|bin|data|dist|docs|latex|main.py|pyproject.toml|python|tests|tmp' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "python"
	touch $@
