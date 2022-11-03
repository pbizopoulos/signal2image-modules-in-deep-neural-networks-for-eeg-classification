.POSIX:

.PHONY: all check clean help

bib_file_name = ms.bib
bib_target = $$(test -s $(bib_file_name) && printf 'bin/check-bib')
tex_file_name = ms.tex
tex_target = $$(test -s $(tex_file_name) && printf 'bin/check-tex')
work_dir = /work

all: bin/all

check: bin/check

clean:
	rm -rf bin/

help:
	@printf 'make all	# Build binaries.\n'
	@printf 'make check	# Check code.\n'
	@printf 'make clean	# Remove binaries.\n'
	@printf 'make help	# Show help.\n'

$(bib_file_name):
	touch $(bib_file_name)

$(tex_file_name):
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(tex_file_name)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/all: $(bib_file_name) $(tex_file_name) .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=bin/ $(tex_file_name)
	touch bin/all

bin/check: .dockerignore .gitignore bin
	$(MAKE) $(bib_target) $(tex_target)

bin/check-bib: $(bib_file_name) .dockerignore .gitignore bin/all
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		checkcites bin/ms.aux'
	touch bin/check-bib

bin/check-tex: $(tex_file_name) .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		chktex $(tex_file_name) && \
		lacheck $(tex_file_name)'
	touch bin/check-tex
