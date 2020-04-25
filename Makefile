# Requires local installation of requirements.txt and texlive-full.
# It takes two minutes and it populates the figures and table.
local-debug:
	./main.py --debug
	latexmk -pdf -cd paper/ms.tex

# Requires local installation of requirements.txt and texlive-full.
# It takes around a week on a NVIDIA Titan X.
local-full:
	./main.py
	latexmk -pdf -cd paper/ms.tex

# Cleans local build.
local-clean:
	latexmk -C -cd paper/ms.tex
	rm -rf paper/images/ selected_models/
	rm paper/ms.bbl paper/results_table.tex

# Requires local installation of docker.
# It takes two minutes.
docker-debug:
	docker build -t dockercode .
	docker run --rm dockercode python3 main.py --debug

# Requires local installation of docker and nvidia-container-toolkit.
# It takes around a week on a NVIDIA Titan X.
docker-full:
	docker build -t dockercode .
	docker run --gpus all --rm dockercode python3 main.py
