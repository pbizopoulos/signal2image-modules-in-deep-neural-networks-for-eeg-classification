.POSIX:

DEBUG = 1

cmd = node index.js

all: tmp/done

check: .dockerignore .gitignore Dockerfile package.json tmp
	$(MAKE) $$(test -s style.css && printf 'tmp/check-css-done') $$(test -s index.html && printf 'tmp/check-html-done') $$(test -s script.js && printf 'tmp/check-js-done') tmp/check-json-done
	if ls -a | grep -v -E -x '.|..|.dockerignore|.env|.gitignore|CNAME|Dockerfile|Makefile|docs|index.html|index.js|misc|package.json|python|script.js|style.css|tmp' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "docs"

clean:
	rm -rf tmp/

.dockerignore:
	printf '*\n!package.json\n' > $@

.gitignore:
	printf '.env\ntmp/\n' > $@

Dockerfile:
	printf 'FROM node\nWORKDIR /usr/src/app\nRUN apt-get update && apt-get install -y jq\nCOPY package.json /usr/src/\nRUN npm install --omit=dev --prefix /usr/src\nRUN npm install --global $$(jq --raw-output ".devDependencies | to_entries[] | \\"\(.key)@\(.value)\\"" /usr/src/package.json)\n' > $@

index.js:
	printf '"use strict"; const dns = require("dns"); const fs = require("fs"); const https = require("https"); const os = require("os"); const serveHandler = require("serve-handler"); const server = https.createServer({ cert: fs.readFileSync("tmp/fullchain.pem"), key: fs.readFileSync("tmp/privkey.pem"), }, (request, response) => { return serveHandler(request, response); }); if (process.env.DEBUG !== "1") { server.listen(443, "0.0.0.0", () => { dns.lookup(os.hostname(), (err, address) => { console.log(`Container IP address: https://$${address}`); }); process.on("SIGINT", () => { process.exit(0); }); }); }' > $@

tmp:
	mkdir $@

tmp/check-css-done: style.css
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --fix style.css && \
		css-validator --profile css3svg style.css'
	touch $@

tmp/check-html-done: index.html
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --no-preserve-newlines --type html --replace index.html && \
		html-validate index.html'
	touch $@

tmp/check-js-done: index.js script.js
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		rome check --apply-unsafe *.js && \
		rome format --write *.js'
	touch $@

tmp/check-json-done: package.json
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) /bin/sh -c '\
		rome check --apply-unsafe package.json && \
		rome format --write package.json'
	touch $@

tmp/done: index.js tmp tmp/fullchain.pem
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env DEBUG=$(DEBUG) \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) $(cmd)
	touch $@

tmp/fullchain.pem: .dockerignore .gitignore Dockerfile package.json
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app \
		$$(docker image build --quiet .) openssl req -subj "/C=.." -nodes -x509 -keyout tmp/privkey.pem -out $@

package.json:
	printf '{\n\t"dependencies": {\n\t\t"serve": "latest"\n\t},\n\t"devDependencies": {\n\t\t"css-validator": "latest",\n\t\t"html-validate": "latest",\n\t\t"js-beautify": "latest",\n\t\t"rome": "latest",\n\t\t"stylelint": "latest",\n\t\t"stylelint-config-standard": "latest",\n\t\t"stylelint-order": "latest"\n\t},\n\t"stylelint": {\n\t\t"extends": "stylelint-config-standard",\n\t\t"plugins": ["stylelint-order"],\n\t\t"rules": {\n\t\t\t"order/properties-alphabetical-order": true\n\t\t}\n\t}\n}\n' > $@
