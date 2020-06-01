PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app

all:
	make clean
	make code
	make paper

code:
	make venv
	. venv/bin/activate; ./main.py $(ARGS)

clean-code:
	rm -rf tmp/ __pycache__/ venv/

venv: requirements.txt
	python -m venv venv
	. venv/bin/activate; pip install -U pip; pip install -Ur requirements.txt

paper:
	latexmk -gg -pdf -quiet -cd paper/ms.tex

clean-paper:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +

clean:
	make clean-code
	make clean-paper

docker:
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=/usr/src/app/tmp \
		-e TORCH_HOME=/usr/src/app/tmp \
		-v $(PWD):$(WORKDIR) \
		$(GPU) $(PROJECT) \
		python3 main.py $(ARGS)
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/paper/:/doc/ \
		thomasweise/docker-texlive-full \
		latexmk -gg -pdf -quiet -cd /doc/ms.tex

.PHONY: paper
