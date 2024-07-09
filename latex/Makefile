.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf tmp/

.gitignore:
	printf '.env\ntmp/\n' > $@

flake.nix:
	nix flake init

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.nix ms.bib ms.tex
	nix develop --command sh -c '\
            set -e && \
            latexmk -outdir=tmp/ -pdf ms.tex && \
            touch tmp/ms.bbl && \
            cp tmp/ms.bbl . && \
            tar cf tmp/tex.tar ms.bbl ms.bib ms.tex $(grep "^INPUT ./" tmp/ms.fls | uniq | cut -b 9-) && \
            rm ms.bbl'
	touch $@

tmp/check-done: .gitignore flake.nix ms.bib ms.tex
	nix develop .#check --command sh -c "\
            set -e && \
	    nixfmt flake.nix && \
	    nix flake check && \
            checkcites tmp/ms.aux && \
            chktex ms.tex && \
            lacheck ms.tex && \
            if ls -ap | grep -v -E -x './|../|.gitignore|Makefile|flake.lock|flake.nix|ms.bib|ms.tex|prm/|python/|tmp/' | grep -q .; then exit 1; fi && \
            test $$(basename $$(pwd)) = 'latex'"
	touch $@
