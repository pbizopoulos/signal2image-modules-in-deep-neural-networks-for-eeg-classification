PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app

local:
	./main.py ${ARGS}
	latexmk -pdf -quiet -cd paper/ms.tex

docker:
	docker build -t ${PROJECT} .
	docker run --rm --user $(shell id -u):$(shell id -g) -v ${PWD}:${WORKDIR} ${GPU} ${PROJECT} python3 main.py ${ARGS}
	docker run --rm --user $(shell id -u):$(shell id -g) -v ${PWD}/paper/:/doc/ thomasweise/docker-texlive-full latexmk -pdf -quiet -cd /doc/ms.tex

clean:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +
	rm -rf tmp/ __pycache__/
