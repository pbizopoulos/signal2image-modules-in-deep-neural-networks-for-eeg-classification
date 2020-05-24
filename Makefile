PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app

local:
	make clean-paper
	make venv
	. venv/bin/activate; ./main.py $(ARGS)
	latexmk -gg -pdf -quiet -cd paper/ms.tex

venv: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt

docker:
	docker build -t $(PROJECT) .
	docker run --rm --user $(shell id -u):$(shell id -g) -v $(PWD):$(WORKDIR) $(GPU) $(PROJECT) python3 main.py $(ARGS)
	docker run --rm --user $(shell id -u):$(shell id -g) -v $(PWD)/paper/:/doc/ thomasweise/docker-texlive-full latexmk -gg -pdf -quiet -cd /doc/ms.tex

clean-paper:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +

clean:
	make clean-paper
	rm -rf tmp/ __pycache__/
	rm -rf venv/
