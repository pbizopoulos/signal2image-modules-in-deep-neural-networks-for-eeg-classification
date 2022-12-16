.POSIX:

.PHONY: all check clean

DEBUG = 1

css_file_name = style.css
css_target = $$(test -s $(css_file_name) && printf 'bin/check-css')
html_file_name = index.html
html_target = $$(test -s $(html_file_name) && printf 'bin/check-html')
interactive_tty_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
make_all_docker_cmd = /bin/sh -c "serve $$(test $(DEBUG) = 1 && printf '& sleep 1; kill $$!')"
js_file_name = script.js
js_target = $$(test -s $(js_file_name) && printf 'bin/check-js')

all: $(html_file_name) .dockerignore Dockerfile package.json
	docker container run \
		$(interactive_tty_arg) \
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

bin/check-css: $(css_file_name) .dockerignore .gitignore Dockerfile bin bin/stylelintrc.json package.json
	docker container run \
		$(interactive_tty_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		stylelint --config bin/stylelintrc.json --fix $(css_file_name) && \
		css-validator --profile css3svg $(css_file_name)'
	touch bin/check-css

bin/check-html: $(html_file_name) .dockerignore .gitignore Dockerfile bin package.json
	docker container run \
		$(interactive_tty_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace $(html_file_name) && \
		html-validate $(html_file_name)'
	touch bin/check-html

bin/check-js: $(js_file_name) .dockerignore .gitignore Dockerfile bin bin/eslintrc.js package.json
	docker container run \
		$(interactive_tty_arg) \
		--rm \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) eslint --config bin/eslintrc.js --fix $(js_file_name)
	touch bin/check-js

bin/eslintrc.js: bin
	printf 'module.exports = { "env": { "browser": true, "es2021": true }, "extends": "eslint:recommended", "overrides": [ ], "parserOptions": { "ecmaVersion": "latest" }, "rules": { "indent": [ "error", "tab", { "SwitchCase": 1 } ], "linebreak-style": [ "error", "unix" ], "no-multiple-empty-lines": [ "error", { "max": 1 } ], "padding-line-between-statements": [ "error", { blankLine: "always", prev: "*", next: "function" }, { blankLine: "always", prev: "function", next: "*" } ], "quotes": [ "error", "single" ], "semi": [ "error", "always" ], "sort-keys": "error", "no-undef": 0 } }\n' > bin/eslintrc.js

bin/stylelintrc.json: bin
	printf '{ "extends": "stylelint-config-standard", "plugins": [ "stylelint-order" ], "rules": { "indentation": "tab", "order/properties-alphabetical-order": true } }\n' > bin/stylelintrc.json

package.json:
	printf '{ "devDependencies": { "css-validator": "latest", "eslint": "latest", "html-validate": "latest", "js-beautify": "latest", "serve": "latest", "stylelint": "latest", "stylelint-config-standard": "latest", "stylelint-order": "latest" } }' > package.json
