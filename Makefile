.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf tmp/

.gitignore:
	printf 'tmp/\n' > $@

tmp:
	mkdir $@

tmp/all-done:
	touch $@

tmp/check-done: .gitignore README
	nix run github:pbizopoulos/check-readme?dir=python README
	test -s .github/workflows/workflow.yml && nix run nixpkgs#actionlint .github/workflows/workflow.yml && nix run nixpkgs#yamlfmt .github/workflows/workflow.yml
	test -s *.sh && nix run nixpkgs#shellcheck *.sh && nix run nixpkgs#shfmt -- --posix --write *.sh || true
	if ls -ap | grep -v -E -x './|../|.deploy.sh|.deploy-requirements.sh|.env|.git/|.github/|.gitignore|CITATION|LICENSE|Makefile|README|docs/|latex/|nix/|python/|tmp/' | grep -q .; then false; fi
	if echo "$$(basename $$(pwd))" | grep -v -E -x '^[a-z0-9]+([-.][a-z0-9]+)*$$'; then false; fi
	touch $@
