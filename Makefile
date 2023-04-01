.POSIX:

make_all_docker_cmd = /bin/sh -c 'latexmk -outdir=bin/ -pdf ms.tex && touch bin/ms.bbl && cp bin/ms.bbl . && tar cf bin/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" bin/ms.fls | uniq | cut -b 9-) && rm ms.bbl'

all: bin/done

check:
	mkdir -p bin/check
	$(MAKE) bin/done $$(test -s ms.bib && printf 'bin/check/bib-done') bin/check/tex-done

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n' > $@

.gitignore:
	printf 'bin/\n' > $@

Dockerfile:
	printf 'FROM texlive/texlive\nWORKDIR /usr/src/app\nRUN apt-get update && apt-get install -y python3-pip\nRUN python3 -m pip install --break-system-packages rebiber\n' > $@

bin:
	mkdir $@

bin/done: .dockerignore .gitignore Dockerfile bin ms.bib ms.tex
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/check/bib-done: .dockerignore Dockerfile ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		checkcites bin/ms.aux && \
		rebiber --input_bib ms.bib --remove url'
	touch $@

bin/check/tex-done: .dockerignore Dockerfile ms.tex
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		chktex ms.tex && \
		lacheck ms.tex'
	touch $@

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@
