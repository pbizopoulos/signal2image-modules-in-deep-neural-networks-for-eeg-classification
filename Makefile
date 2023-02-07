.POSIX:

make_all_docker_cmd = /bin/sh -c 'latexmk -gg -pdf -outdir=bin/ ms.tex && tar cf bin/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" bin/ms.fls | uniq | cut -b 9-)'

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
	printf 'FROM texlive/texlive\n' > $@

bin:
	mkdir $@

bin/all: .dockerignore .gitignore Dockerfile bin ms.bib ms.tex
	touch bin/ms.bbl && cp bin/ms.bbl .
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	rm ms.bbl
	touch $@

bin/check-bib: .dockerignore Dockerfile ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) checkcites bin/ms.aux
	docker container run \
		--env HOME=/usr/src/app/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		python /bin/sh -c "\
		python3 -m pip install --upgrade pip && \
		python3 -m pip install rebiber && \
		bin/.local/bin/rebiber --input_bib ms.bib --sort True"
	docker container run \
		--env HOME=/usr/src/app/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		node npm exec --yes -- git+https://github.com/FlamingTempura/bibtex-tidy.git --curly --tab --no-align --blank-lines --duplicates=key --sort-fields ms.bib
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
