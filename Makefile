ms.pdf: results ms.tex ms.bib
	latexmk -gg -pdf -quiet ms.tex

results: venv $(shell find . -maxdepth 1 -name '*.py')
	. venv/bin/activate; ./main.py $(ARGS)
	touch -c results

venv: requirements.txt
	python3 -m venv venv
	. venv/bin/activate; pip install -U pip wheel; pip install -Ur requirements.txt
	touch -c venv

clean:
	rm -rf __pycache__/ cache/ venv/ arxiv.tar results/ ms.bbl
	latexmk -C ms.tex

docker-ms.pdf:
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/:/home/latex \
		aergus/latex \
		latexmk -gg -pdf -quiet -cd /home/latex/ms.tex

PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app
docker:
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=$(WORKDIR)/cache \
		-e TORCH_HOME=$(WORKDIR)/cache \
		-v $(PWD):$(WORKDIR) \
		$(PROJECT) \
		./main.py $(ARGS)
	make docker-ms.pdf

docker-gpu:
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=$(WORKDIR)/cache \
		-e TORCH_HOME=$(WORKDIR)/cache \
		-v $(PWD):$(WORKDIR) \
		--gpus all $(PROJECT) \
		./main.py $(ARGS)
	make docker-ms.pdf

arxiv:
	curl -LO https://arxiv.org/e-print/$(ARXIV_ID)
	tar -xvf $(ARXIV_ID)
	docker build -t $(PROJECT)-arxiv .
	make docker-ms.pdf
	rm $(ARXIV_ID)

arxiv.tar:
	tar -cvf arxiv.tar ms.tex ms.bib ms.bbl results/*.{pdf,tex}

TAG=results
upload-results:
	hub release create -m 'Results release' $(TAG)
	for f in $(shell ls results/*); do hub release edit -m 'Results' -a $$f $(TAG); done

download-results:
	mkdir -p results ; cd results ; hub release download $(TAG) ; cd ..

delete-results:
	hub release delete $(TAG)
	git push origin :$(TAG)
