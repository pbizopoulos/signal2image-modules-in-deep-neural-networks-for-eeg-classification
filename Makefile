.POSIX:

CACHE_DIR = cache
DOCKER_WORKDIR = /usr/src/app
NAME_CURRENT_DIR = $(notdir $(shell pwd))
RELEASE_NAME = v1
RESULTS_DIR = results
ROOT_CODE = main.py
ROOT_TEX_NO_EXT = ms
SRC_CODE = $(shell find . -maxdepth 1 -name '*.py')

$(ROOT_TEX_NO_EXT).pdf: $(ROOT_TEX_NO_EXT).tex $(ROOT_TEX_NO_EXT).bib $(RESULTS_DIR)
	latexmk -gg -pdf -quiet $<

$(RESULTS_DIR): venv $(SRC_CODE)
	rm -rf $@/
	. $</bin/activate; python3 $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)

venv: requirements.txt
	rm -rf $@/
	python3 -m $@ $@/
	. $@/bin/activate; pip install -U pip wheel; pip install -Ur $<

clean:
	rm -rf __pycache__/ $(CACHE_DIR)/ $(RESULTS_DIR)/ venv/ arxiv.tar $(ROOT_TEX_NO_EXT).bbl
	latexmk -C $(ROOT_TEX_NO_EXT)

docker:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/$(CACHE_DIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		$(NAME_CURRENT_DIR) \
		python3 $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)
	make docker-pdf

docker-gpu:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-e HOME=$(DOCKER_WORKDIR)/$(CACHE_DIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		--gpus all $(NAME_CURRENT_DIR) \
		python3 $(ROOT_CODE) $(ARGS) --cache-dir $(CACHE_DIR) --results-dir $(RESULTS_DIR)
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
	docker build -t $(NAME_CURRENT_DIR)-arxiv .
	make docker-pdf
	rm $(ARXIV_ID)

arxiv.tar:
	tar -cvf arxiv.tar $(ROOT_TEX_NO_EXT).{tex,bib,bbl} $(RESULTS_DIR)/*.{pdf,tex}

upload-results:
	hub release create -m 'Results release' $(RELEASE_NAME)
	for f in $(shell ls $(RESULTS_DIR)/*); do hub release edit -m 'Results' -a $$f $(RELEASE_NAME); done

download-results:
	mkdir -p $(RESULTS_DIR) ; cd $(RESULTS_DIR) ; hub release download $(RELEASE_NAME) ; cd ..

delete-results:
	hub release delete $(RELEASE_NAME)
	git push origin :$(RELEASE_NAME)
