.POSIX:

.PHONY: all check clean

aux_file_name = ms.aux
bbl_file_name = ms.bbl
bib_file_name = ms.bib
bib_target = $$(test -s $(bib_file_name) && printf 'bin/check-bib')
fls_file_name = ms.fls
interactive_tty_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
make_all_docker_cmd = /bin/sh -c 'latexmk -gg -pdf -outdir=bin/ $(tex_file_name) && tar cf bin/tex.tar $(bbl_file_name) $(bib_file_name) $(tex_file_name) $$(grep "^INPUT ./" bin/$(fls_file_name) | uniq | cut -b 9-)'
tex_file_name = ms.tex

all: bin/all

check:
	$(MAKE) $(bib_target) bin/check-tex

clean:
	rm -rf bin/

$(bib_file_name):
	touch $(bib_file_name)

$(tex_file_name):
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(tex_file_name)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

Dockerfile:
	printf 'FROM texlive/texlive\n' > Dockerfile

bin:
	mkdir bin

bin/all: $(bib_file_name) $(tex_file_name) .dockerignore .gitignore Dockerfile bin
	touch bin/$(bbl_file_name) && cp bin/$(bbl_file_name) .
	docker container run \
		$(interactive_tty_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	rm $(bbl_file_name)
	touch bin/all

bin/check-bib: bin/all
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) checkcites bin/$(aux_file_name)
	docker container run \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		python /bin/sh -c "\
		python3 -m pip install --upgrade pip && \
		python3 -m pip install rebiber && \
		bin/.local/bin/rebiber --input_bib $(bib_file_name) --sort True"
	docker container run \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		node npm exec --yes -- git+https://github.com/FlamingTempura/bibtex-tidy.git --curly --tab --no-align --blank-lines --duplicates=key --sort-fields $(bib_file_name)
	touch bin/check-bib

bin/check-tex: $(tex_file_name) .dockerignore Dockerfile
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		chktex $(tex_file_name) && \
		lacheck $(tex_file_name)'
	touch bin/check-tex
