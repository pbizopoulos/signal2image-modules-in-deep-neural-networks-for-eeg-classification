.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf result tmp/

.gitignore:
	printf '.env\nresult\ntmp/\n' > $@

flake.nix:
	nix flake init

index.html:
	touch $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.nix index.html
	nix develop --command sh -c "\
            [ ! -z $$STAGE ] && openssl req -subj '/C=..' -nodes -x509 -keyout tmp/privkey.pem -out tmp/fullchain.pem && http-server --tls --cert tmp/fullchain.pem --key tmp/privkey.pem || true"
	touch $@

tmp/check-done: .gitignore flake.nix index.html
	nix develop .#check --command sh -c "\
            set -e && \
            nixfmt flake.nix && \
	    nix flake check && \
            [ -e index.html ] && js-beautify --end-with-newline --indent-inner-html --no-preserve-newlines --type html --replace index.html && \
            [ -e script.js ] && biome check --unsafe --write script.js || true && \
            if ls -ap | grep -v -E -x './|../|.env|.gitignore|CNAME|Makefile|index.html|index.js|flake.lock|flake.nix|prm/|pyscript/|python/|script.js|style.css|tmp/' | grep -q .; then exit 1; fi && \
            test $$(basename $$(pwd)) = 'docs'"
	touch $@
