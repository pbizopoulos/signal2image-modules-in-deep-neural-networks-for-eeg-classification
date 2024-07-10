.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf result tmp/

.gitignore:
	printf '.env\nresult\ntmp/\n' > $@

flake.nix:
	nix flake init

main.py:
	touch $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.nix main.py
	nix develop --command sh -c "\
		set -e && \
		PYTHONDONTWRITEBYTECODE=1 python3 main.py && \
		exit"
	touch $@

tmp/check-done: .gitignore flake.nix main.py
	nix develop .#check --command sh -c "\
		set -e && \
		export PYTHONDONTWRITEBYTECODE=1 && \
		nixfmt flake.nix && \
		nix flake check && \
		check-python-script main.py && \
		ruff format --cache-dir tmp/ruff main.py && \
		ruff check --cache-dir tmp/ruff --exit-non-zero-on-fix --fix --select ALL --unsafe-fixes main.py && \
		mypy --cache-dir tmp/mypy --ignore-missing-imports --strict main.py && \
		if [ -d 'templates/' ]; then djlint templates/ --lint --profile=jinja --quiet --reformat; fi && \
		[ -z $$STAGE ] && exit 0 || (unset STAGE && pydoc -w main && mv main.html tmp/) && \
		if ls -ap | grep -v -E -x './|../|.env|.gitignore|Makefile|flake.lock|flake.nix|main.py|prm/|python/|result|static/|templates/|tmp/' | grep -q .; then exit 1; fi && \
		test $$(basename $$(pwd)) = 'python'"
	touch $@
