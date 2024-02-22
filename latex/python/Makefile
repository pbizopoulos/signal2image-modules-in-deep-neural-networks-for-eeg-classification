.POSIX:

cmd = python3 main.py

all: tmp tmp/all-done

check: tmp tmp/check-done

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
	printf '[project]\nname = "UNKNOWN"\nversion = "0.0.0"\n\n[project.optional-dependencies]\ndev = ["coverage","djlint","mypy","ruff","ssort","toml-sort"]' > $@

tmp:
	mkdir $@

tmp/all-done: .dockerignore .gitignore Dockerfile main.py pyproject.toml
	docker container run \
		$$(command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all') \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env HOME=/usr/src/app/tmp \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--env STAGING=$(STAGING) \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(cmd)
	touch $@

tmp/check-done: .dockerignore .gitignore Dockerfile main.py pyproject.toml
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--env HOME=/usr/src/app/tmp \
		--env PYTHONDONTWRITEBYTECODE=1 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'toml-sort --all --in-place pyproject.toml && ssort main.py && ruff format --cache-dir tmp/ruff main.py && ruff --cache-dir tmp/ruff --fix --ignore D --select ALL --unsafe-fixes main.py && mypy --cache-dir tmp/mypy --ignore-missing-imports --install-types --non-interactive --strict main.py && if [ -d "templates/" ]; then djlint templates/ --lint --profile=jinja --quiet --reformat; fi && ([ -z $(STAGING) ] && exit 0 || (unset STAGING && coverage run --data-file=tmp/.coverage main.py && coverage html --data-file=tmp/.coverage --directory tmp/ --ignore-errors))'
	if ls -ap | grep -v -E -x './|../|.dockerignore|.env|.gitignore|Dockerfile|Makefile|main.py|prm/|pyproject.toml|python/|static/|templates/|tmp/' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "python"
	touch $@
