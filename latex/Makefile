.POSIX:

cmd = /bin/sh -c 'latexmk -outdir=bin/ -pdf ms.tex && touch bin/ms.bbl && cp bin/ms.bbl . && tar cf bin/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" bin/ms.fls | uniq | cut -b 9-) && rm ms.bbl'

all: bin bin/done

check: bin/done .dockerignore .gitignore Dockerfile bin bin/check bin/check/bib-done bin/check/tex-done
	if ls -a | grep -v -E -x '.|..|.dockerignore|.gitignore|Dockerfile|Makefile|bin|dist|docs|latex|ms.bib|ms.tex|python' | grep -q .; then false; fi

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n' > $@

.gitignore:
	printf 'bin/\n' > $@

Dockerfile:
	printf 'FROM texlive/texlive\nRUN apt-get update && apt-get install -y python3-pip\nRUN python3 -m pip install --break-system-packages rebiber\n' > $@

bin:
	mkdir $@

bin/check:
	mkdir $@

bin/check/bib-done: ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		checkcites bin/ms.aux && \
		rebiber --input_bib ms.bib --remove url'
	touch $@

bin/check/tex-done: ms.tex
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		chktex ms.tex && \
		lacheck ms.tex'
	touch $@

bin/done: .dockerignore .gitignore Dockerfile ms.bib ms.tex
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
