local:
	./main.py $(ARGS)
	latexmk -pdf -quiet -cd paper/ms.tex

clean:
	latexmk -C -cd paper/ms.tex
	rm -rf paper/images/ models/
	rm paper/ms.bbl paper/results_table.tex

docker:
	docker build -t dockercode .
	docker run --rm ${GPU} dockercode python3 main.py ${ARGS}
