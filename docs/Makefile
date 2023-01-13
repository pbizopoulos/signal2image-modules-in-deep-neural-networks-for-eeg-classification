.POSIX:

.PHONY: all check clean

DEBUG = 1

css_file_name = style.css
css_target = $$(test -s $(css_file_name) && printf 'bin/check-css')
html_file_name = index.html
html_target = $$(test -s $(html_file_name) && printf 'bin/check-html')
interactive_tty_docker_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
make_all_docker_cmd = /bin/sh -c "serve --ssl-cert bin/cert.pem --ssl-key bin/key.pem $$(test $(DEBUG) = 1 && printf '& sleep 1; kill $$!')"
js_file_name = script.js
js_target = $$(test -s $(js_file_name) && printf 'bin/check-js')

all: $(html_file_name) .dockerignore Dockerfile bin/cert.pem package.json
	docker container run \
		$(interactive_tty_docker_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--publish 3000:3000 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) $(make_all_docker_cmd)

check:
	$(MAKE) $(css_target) $(html_target) $(js_target)

clean:
	rm -rf bin/

$(html_file_name):
	printf '\n' > $(html_file_name)

.dockerignore:
	printf '*\n!package.json\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

Dockerfile:
	printf 'FROM node\nRUN apt-get update && apt-get install -y jq\nCOPY package.json .\nRUN npm install --global $$(jq --raw-output ".devDependencies | to_entries | map_values( .key + \"@\" + .value ) | join(\" \")" package.json)\n' > Dockerfile

bin:
	mkdir bin

bin/cert.pem: .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		alpine/openssl req -subj "/C=.." -nodes -x509 -keyout bin/key.pem -out bin/cert.pem

bin/check-css: $(css_file_name) .dockerignore .gitignore Dockerfile bin bin/stylelintrc.json package.json
	docker container run \
		$(interactive_tty_docker_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --config bin/stylelintrc.json --fix $(css_file_name) && \
		css-validator --profile css3svg $(css_file_name)'
	touch bin/check-css

bin/check-html: $(html_file_name) .dockerignore .gitignore Dockerfile bin package.json
	docker container run \
		$(interactive_tty_docker_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace $(html_file_name) && \
		html-validate $(html_file_name)'
	touch bin/check-html

bin/check-js: $(js_file_name) .dockerignore .gitignore Dockerfile bin package.json
	docker container run \
		$(interactive_tty_docker_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		rome check --apply-suggested $(js_file_name) && \
		rome format --line-width 320 --quote-style single --write $(js_file_name)'
	touch bin/check-js

bin/stylelintrc.json: bin
	printf '{ "extends": "stylelint-config-standard", "plugins": [ "stylelint-order" ], "rules": { "indentation": "tab", "order/properties-alphabetical-order": true } }\n' > bin/stylelintrc.json

package.json:
	printf '{\n\t"devDependencies": {\n\t\t"css-validator": "latest",\n\t\t"html-validate": "latest",\n\t\t"js-beautify": "latest",\n\t\t"rome": "latest",\n\t\t"serve": "latest",\n\t\t"stylelint": "latest",\n\t\t"stylelint-config-standard": "latest",\n\t\t"stylelint-order": "latest"\n\t}\n}\n' > package.json
