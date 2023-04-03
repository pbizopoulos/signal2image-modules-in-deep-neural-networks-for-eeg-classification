.POSIX:

DEBUG = 1

make_all_docker_cmd = node index.js

all: bin/done

check:
	mkdir -p bin/check
	$(MAKE) $$(test -s style.css && printf 'bin/check/css-done') $$(test -s index.html && printf 'bin/check/html-done') $$(test -s script.js && printf 'bin/check/js-done')

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n!package.json\n' > $@

.gitignore:
	printf 'bin/\n' > $@

Dockerfile:
	printf 'FROM node\nWORKDIR /usr/src\nCOPY package.json .\nRUN npm install --omit=dev\nWORKDIR /usr/src/app\nRUN apt-get update && apt-get install -y jq\nCOPY package.json .\nRUN npm install --global $$(jq -r ".devDependencies | to_entries[] | \\"\(.key)@\(.value)\\"" package.json)\n' > $@

bin:
	mkdir $@

bin/cert.pem: .dockerignore .gitignore Dockerfile bin package.json
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) openssl req -subj "/C=.." -nodes -x509 -keyout bin/key.pem -out $@

bin/check/css-done: .dockerignore .gitignore Dockerfile package.json style.css
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --fix style.css && \
		css-validator --profile css3svg style.css'
	touch $@

bin/check/html-done: .dockerignore .gitignore Dockerfile bin index.html package.json
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace index.html && \
		html-validate index.html'
	touch $@

bin/check/js-done: .dockerignore .gitignore Dockerfile bin package.json script.js
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		rome check --apply-unsafe . && \
		rome format --line-width 320 --write .'
	touch $@

bin/done: bin/cert.pem index.html index.js
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--pid=host \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

index.html:
	touch $@

index.js:
	printf '"use strict";\nconst fs = require("fs");\nconst handler = require("serve-handler");\nconst https = require("https");\n\nconst options = {\n\tkey: fs.readFileSync("bin/key.pem"),\n\tcert: fs.readFileSync("bin/cert.pem"),\n};\n\nconst server = https.createServer(options, (request, response) => {\n\treturn handler(request, response);\n});\n\nif (process.env.DEBUG !== "1") {\n\tserver.listen(8000, "172.17.0.2", () => { console.log("Running at https://172.17.0.2:8000"); });\n}\n' > $@

package.json:
	printf '{\n\t"dependencies": {\n\t\t"serve": "latest"\n\t},\n\t"devDependencies": {\n\t\t"css-validator": "latest",\n\t\t"html-validate": "latest",\n\t\t"js-beautify": "latest",\n\t\t"rome": "latest",\n\t\t"stylelint": "latest",\n\t\t"stylelint-config-standard": "latest",\n\t\t"stylelint-order": "latest"\n\t},\n\t"stylelint": {\n\t\t"extends": "stylelint-config-standard",\n\t\t"plugins": ["stylelint-order"],\n\t\t"rules": {\n\t\t\t"indentation": "tab",\n\t\t\t"order/properties-alphabetical-order": true\n\t\t}\n\t}\n}\n' > $@
