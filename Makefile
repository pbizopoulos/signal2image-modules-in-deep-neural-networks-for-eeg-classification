.POSIX:

DIR_NAME = $(notdir $(shell pwd))
DOCKER_WORKDIR = /usr/src/app
RELEASE_NAME = v1
ROOT_CODE = main.py
ROOT_TEX_NO_EXT = ms
SRC_CODE = $(shell find . -maxdepth 1 -name '*.py')

$(ROOT_TEX_NO_EXT).pdf: $(ROOT_TEX_NO_EXT).tex $(ROOT_TEX_NO_EXT).bib results
	latexmk -gg -pdf -quiet $<

results: venv $(SRC_CODE)
	rm -rf $@/
	. venv/bin/activate; python3 $(ROOT_CODE) $(ARGS)

venv: requirements.txt
	rm -rf $@/
	python3 -m $@ $@/
	. $@/bin/activate; pip install -U pip wheel; pip install -Ur $<

clean:
	rm -rf __pycache__/ cache/ results/ venv/ arxiv.tar $(ROOT_TEX_NO_EXT).bbl
	latexmk -C $(ROOT_TEX_NO_EXT)

docker:
	docker build -t $(DIR_NAME) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/cache \
		-e TORCH_HOME=$(DOCKER_WORKDIR)/cache \
		-v $(PWD):$(DOCKER_WORKDIR) \
		$(DIR_NAME) \
		python3 $(ROOT_CODE) $(ARGS)
	make docker-pdf

docker-gpu:
	docker build -t $(DIR_NAME) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/cache \
		-e TORCH_HOME=$(DOCKER_WORKDIR)/cache \
		-v $(PWD):$(DOCKER_WORKDIR) \
		--gpus all $(DIR_NAME) \
		python3 $(ROOT_CODE) $(ARGS)
	make docker-pdf

docker-pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -gg -pdf -quiet -cd /home/latex/$(ROOT_TEX_NO_EXT).tex

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	docker build -t $(DIR_NAME)-arxiv .
	make docker-pdf
	rm $(ARXIV_ID)

arxiv.tar:
	tar -cvf arxiv.tar $(ROOT_TEX_NO_EXT).{tex,bib,bbl} results/*.{pdf,tex}

upload-results:
	hub release create -m 'Results release' $(RELEASE_NAME)
	for f in $(shell ls results/*); do hub release edit -m 'Results' -a $$f $(RELEASE_NAME); done

download-results:
	mkdir -p results ; cd results ; hub release download $(RELEASE_NAME) ; cd ..

delete-results:
	hub release delete $(RELEASE_NAME)
	git push origin :$(RELEASE_NAME)
