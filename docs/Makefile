.POSIX:

.PHONY: all clean

all: venv/bin/activate
	. venv/bin/activate && python3 main.py

venv/bin/activate: requirements.txt
	python3 -m venv venv
	. venv/bin/activate && python3 -m pip install --upgrade pip wheel && python3 -m pip install -r requirements.txt

clean:
	rm -rf index.html venv/
