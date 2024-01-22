.POSIX:

cmd = node index.js

all: tmp/all-done

check: tmp/check-done

clean:
	rm -rf tmp/

.dockerignore:
	printf '*\n!package.json\n' > $@

.gitignore:
	printf '.env\ntmp/\n' > $@

Dockerfile:
	printf 'FROM node\nRUN apt-get update && apt-get install -y jq\nWORKDIR /usr/src/app\nCOPY package.json /usr/src/\nRUN npm install --omit=dev --prefix /usr/src\nRUN npm install --global $$(jq --raw-output ".devDependencies | to_entries[] | \\"\(.key)@\(.value)\\"" /usr/src/package.json)\n' > $@

index.js:
	printf 'const dns = require("dns"); const fs = require("fs"); const https = require("https"); const os = require("os"); const serveHandler = require("serve-handler"); const server = https.createServer({ cert: fs.readFileSync("tmp/fullchain.pem"), key: fs.readFileSync("tmp/privkey.pem"), }, (request, response) => { return serveHandler(request, response); }); if (process.env.STAGING) { server.listen(443, "0.0.0.0", () => { dns.lookup(os.hostname(), (err, address) => { console.log(`Container IP address: https://$${address}`); }); process.on("SIGINT", () => { process.exit(0); }); }); }' > $@

package.json:
	printf '{"stylelint":{"extends":"stylelint-config-standard","plugins":["stylelint-order"],"rules":{"order/properties-alphabetical-order":true}},"dependencies":{"serve":"latest"},"devDependencies":{"@biomejs/biome":"latest","css-validator":"latest","html-validate":"latest","js-beautify":"latest","sort-package-json":"latest","stylelint":"latest","stylelint-config-standard":"latest","stylelint-order":"latest"}}' > $@

tmp:
	mkdir $@

tmp/all-done: index.js tmp tmp/fullchain.pem
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env STAGING=$(STAGING) \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(cmd)
	touch $@

tmp/check-css-done: style.css
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --fix style.css && \
		css-validator --profile css3svg style.css'
	touch $@

tmp/check-done: .dockerignore .gitignore Dockerfile package.json tmp
	$(MAKE) $$(test -s style.css && printf 'tmp/check-css-done') $$(test -s index.html && printf 'tmp/check-html-done') tmp/check-js-done
	if ls -ap | grep -v -E -x './|../|.dockerignore|.env|.gitignore|CNAME|Dockerfile|Makefile|docs/|index.html|index.js|package.json|prm/|python/|script.js|style.css|tmp/' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "docs"
	touch $@

tmp/check-html-done: index.html
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --no-preserve-newlines --type html --replace index.html && \
		html-validate index.html'
	touch $@

tmp/check-js-done: index.js
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c "\
		script_js=$$([ -e script.js ] && echo script.js) && \
		style_css=$$([ -e style.css ] && echo style.css) && \
		sort-package-json package.json && \
		biome check --apply-unsafe index.js package.json \$$script_js \$$style_css"
	touch $@

tmp/fullchain.pem: .dockerignore .gitignore Dockerfile package.json
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) openssl req -subj "/C=.." -nodes -x509 -keyout tmp/privkey.pem -out $@
