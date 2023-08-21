.POSIX:

DEBUG = 1

cmd = python3 main.py

all: tmp/done

check: tmp/check-done

clean:
	rm -rf tmp/

.dockerignore:
	printf '*\n!pyproject.toml\n' > $@

.gitignore:
	printf '.env\ntmp/\n' > $@

Dockerfile:
	printf 'FROM python\nWORKDIR /usr/src/app\nCOPY pyproject.toml .\nRUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]\n' > $@

main.py:
	touch $@

pyproject.toml:
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\n\n[project.optional-dependencies]\ndev = [\n\t"black",\n\t"djlint",\n\t"mypy",\n\t"ruff",\n\t"ssort",\n]\n' > $@

tmp:
	mkdir $@

tmp/done: .dockerignore .gitignore Dockerfile main.py pyproject.toml tmp
	docker container run \
		$$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all') \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=/usr/src/app/tmp \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--env PYTHONOPTIMIZE=$(shell [ "$(DEBUG)" = "0" ] && echo "1" || echo "0") \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(cmd)
	touch $@

tmp/check-done: .dockerignore .gitignore Dockerfile main.py pyproject.toml tmp
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--env HOME=/usr/src/app/tmp \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'ssort main.py && black main.py && ruff --cache-dir tmp/ruff --fix --ignore D,S101 --select ALL main.py && mypy --cache-dir tmp/mypy --ignore-missing-imports --install-types --non-interactive --strict main.py && if [ -d "assets/templates/" ]; then djlint assets/templates/ --lint --profile=jinja --quiet --reformat; fi'
	if ls -a | grep -v -E -x '.|..|.dockerignore|.env|.gitignore|__init__.py|Dockerfile|Makefile|assets|dist|docs|latex|main.py|pyproject.toml|python|tests|tmp' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "python"
	touch $@
