.POSIX:

make_all_docker_cmd = /bin/sh -c 'latexmk -outdir=bin/ -pdf ms.tex && touch bin/ms.bbl && cp bin/ms.bbl . && tar cf bin/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" bin/ms.fls | uniq | cut -b 9-) && rm ms.bbl'

all: bin/all

check:
	$(MAKE) bin/all $$(test -s ms.bib && printf 'bin/check-bib') bin/check-tex

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n' > $@

.gitignore:
	printf 'bin/\n' > $@

Dockerfile:
	printf 'FROM texlive/texlive\nWORKDIR /usr/src/app\nRUN apt-get update -y && apt-get install -y python3-pip\nRUN python3 -m pip install --break-system-packages rebiber\n' > $@

bin:
	mkdir $@

bin/all: .dockerignore .gitignore Dockerfile bin ms.bib ms.tex
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/check-bib: .dockerignore Dockerfile ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		checkcites bin/ms.aux && \
		rebiber --input_bib ms.bib'
	touch $@

bin/check-tex: .dockerignore Dockerfile ms.tex
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		chktex ms.tex && \
		lacheck ms.tex'
	touch $@

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@
