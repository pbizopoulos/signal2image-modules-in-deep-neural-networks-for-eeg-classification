.POSIX:

cmd = /bin/sh -c 'latexmk -outdir=tmp/ -pdf ms.tex && touch tmp/ms.bbl && cp tmp/ms.bbl . && tar cf bin/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" tmp/ms.fls | uniq | cut -b 9-) && rm ms.bbl && mv tmp/ms.pdf bin/ms.pdf'

all: tmp tmp/done

check: tmp/done .dockerignore .gitignore Dockerfile bin tmp tmp/check tmp/check/bib-done tmp/check/tex-done
	if ls -a | grep -v -E -x '.|..|.dockerignore|.gitignore|Dockerfile|Makefile|assets|bin|dist|docs|latex|ms.bib|ms.tex|python|tmp' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "latex"

clean: mostlyclean

mostlyclean:
	rm -rf bin/ tmp/

.dockerignore:
	printf '*\n' > $@

.gitignore:
	printf 'bin/\ntmp/\n' > $@

Dockerfile:
	printf 'FROM texlive/texlive\nRUN apt-get update && apt-get install -y python3-pip\nRUN python3 -m pip install --break-system-packages rebiber\n' > $@

bin:
	mkdir $@

tmp:
	mkdir $@

tmp/check:
	mkdir $@

tmp/check/bib-done: ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		checkcites tmp/ms.aux && \
		rebiber --input_bib ms.bib --remove url'
	touch $@

tmp/check/tex-done: ms.tex
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		chktex ms.tex && \
		lacheck ms.tex'
	touch $@

tmp/done: .dockerignore .gitignore Dockerfile bin ms.bib ms.tex
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) $(cmd)
	touch $@

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@
