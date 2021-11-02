all: venv/bin/activate
	. venv/bin/activate && python3 -m pip install --upgrade pip wheel && python3 -m pip install -r requirements.txt && python3 main.py

venv/bin/activate:
	python3 -m venv venv

clean:
	rm -rf index.html venv/
