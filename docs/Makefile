.POSIX:

DEBUG = 1

make_all_docker_cmd = /bin/sh -c "serve --ssl-cert bin/cert.pem --ssl-key bin/key.pem $$(test $(DEBUG) = 1 && printf '& sleep 1; kill $$!')"

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
	printf 'FROM node\nRUN apt-get update && apt-get install -y jq\nCOPY package.json .\nRUN npm install --global $$(jq --raw-output ".devDependencies | to_entries | map_values( .key + \\"@\\" + .value ) | join(\\" \\")" package.json)\n' > $@

bin:
	mkdir $@

bin/done: Dockerfile bin/cert.pem index.html package.json
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)
	touch $@

bin/cert.pem: .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		alpine/openssl req -subj "/C=.." -nodes -x509 -keyout bin/key.pem -out $@

bin/check/css-done: .dockerignore .gitignore Dockerfile package.json style.css
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --fix style.css && \
		css-validator --profile css3svg style.css'
	touch $@

bin/check/html-done: .dockerignore .gitignore Dockerfile bin index.html package.json
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace index.html && \
		html-validate index.html'
	touch $@

bin/check/js-done: .dockerignore .gitignore Dockerfile bin package.json script.js
	docker container run \
		--rm \
		--volume $$(pwd):/usr/src/app/ \
		--workdir /usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		rome check --apply-suggested script.js && \
		rome format --line-width 320 --write script.js'
	touch $@

index.html:
	printf '\n' > $@

package.json:
	printf '{\n\t"devDependencies": {\n\t\t"css-validator": "latest",\n\t\t"html-validate": "latest",\n\t\t"js-beautify": "latest",\n\t\t"rome": "latest",\n\t\t"serve": "latest",\n\t\t"stylelint": "latest",\n\t\t"stylelint-config-standard": "latest",\n\t\t"stylelint-order": "latest"\n\t},\n\t"stylelint": {\n\t\t"extends": "stylelint-config-standard",\n\t\t"plugins": ["stylelint-order"],\n\t\t"rules": {\n\t\t\t"indentation": "tab",\n\t\t\t"order/properties-alphabetical-order": true\n\t\t}\n\t}\n}\n' > $@
