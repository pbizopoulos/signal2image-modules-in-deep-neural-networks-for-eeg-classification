all:
	make clean
	make results
	make ms.pdf

clean:
	rm -rf __pycache__/ cache/ venv/ upload_to_arxiv.tar
	make clean-results

clean-results:
	latexmk -C ms.tex
	rm -rf results/ ms.bbl

results: $(shell find . -maxdepth 1 -name '*.py')
	make venv
	. venv/bin/activate; ./main.py $(ARGS)
	touch -c results

venv: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install -U pip wheel; pip install -Ur requirements.txt
	touch -c venv

ms.pdf: ms.tex ms.bib
	latexmk -gg -pdf -quiet ms.tex

view:
	xdg-open ms.pdf

docker-ms.pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/doc/ \
		thomasweise/docker-texlive-full \
		latexmk -gg -pdf -quiet -cd /doc/ms.tex

PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app
GPU != if [[ "$(ARGS)" == *"--gpu"* ]]; then echo "--gpus=all"; fi
docker:
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=$(WORKDIR)/cache \
		-e TORCH_HOME=$(WORKDIR)/cache \
		-v $(PWD):$(WORKDIR) \
		$(GPU) $(PROJECT) \
		./main.py $(ARGS)
	make docker-ms.pdf

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	docker build -t $(PROJECT)-arxiv .
	make docker-ms.pdf
	rm $(ARXIV_ID)

arxiv-tar:
	tar -cvf upload_to_arxiv.tar ms.tex ms.bib ms.bbl results/
