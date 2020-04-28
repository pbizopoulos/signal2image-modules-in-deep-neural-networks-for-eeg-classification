PROJECT=$(notdir $(shell pwd))
WORKDIR=/usr/src/app

local:
	./main.py ${ARGS}
	latexmk -pdf -quiet -cd paper/ms.tex

clean:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +
	rm -rf tmp/ __pycache__/

docker:
	docker build -t ${PROJECT} .
	docker run --rm --user=1000 -v ${PWD}:${WORKDIR} ${GPU} ${PROJECT} python3 main.py ${ARGS}
	docker run --rm --user=1000 -v ${PWD}/paper/:/doc/ thomasweise/docker-texlive-full latexmk -pdf -quiet -cd /doc/ms.tex
