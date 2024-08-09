.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf result tmp/

.gitignore:
	printf '.env\nresult\ntmp/\n' > $@

flake.nix:
	nix flake init

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.lock flake.nix ms.bib ms.tex
	nix develop --ignore-environment .#all
	touch $@

tmp/check-done: .gitignore flake.lock flake.nix ms.bib ms.tex
	nix develop --ignore-environment .#check
	touch $@
