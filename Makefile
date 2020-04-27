PROJECT=signal2image

local:
	./main.py ${ARGS}
	latexmk -pdf -quiet -cd paper/ms.tex

clean:
	find paper/* ! -name ms.tex ! -name ms.bib -type d,f -exec rm -rf {} +
	rm -rf tmp/ __pycache__/

docker:
	docker build -t ${PROJECT} .
	docker run --rm ${GPU} ${PROJECT} python3 main.py ${ARGS}
