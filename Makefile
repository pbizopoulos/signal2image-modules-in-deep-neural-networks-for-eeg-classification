PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app
GPU != if [[ "$(ARGS)" == *"--gpu"* ]]; then echo "--gpus=all"; fi

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
	. venv/bin/activate; pip install -U pip wheel; pip install -Ur requirements.txt

paper:
	latexmk -gg -pdf -quiet -cd paper/ms.tex

clean-paper:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +

clean:
	make clean-code
	make clean-paper

docker:
	make clean
	docker build -t $(PROJECT) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(WORKDIR) \
		-e HOME=$(WORKDIR)/tmp \
		-e TORCH_HOME=$(WORKDIR)/tmp \
		-v $(PWD):$(WORKDIR) \
		$(GPU) $(PROJECT) \
		./main.py $(ARGS)
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD)/paper/:/doc/ \
		thomasweise/docker-texlive-full \
		latexmk -gg -pdf -quiet -cd /doc/ms.tex

.PHONY: paper
