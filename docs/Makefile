.POSIX:

.PHONY: all check clean

DEBUG = 1

css_file_name = style.css
css_target = $$(test -s $(css_file_name) && printf 'bin/check-css')
html_file_name = index.html
html_target = $$(test -s $(html_file_name) && printf 'bin/check-html')
interactive_tty_arg = $$(test -t 0 && printf '%s' '--interactive --tty')
js_file_name = script.js
js_target = $$(test -s $(js_file_name) && printf 'bin/check-js')
npx_timeout_command = $$(test $(DEBUG) = 1 && printf '& sleep 1; kill $$!')

all: .dockerignore Dockerfile
	docker container run \
		$(interactive_tty_arg) \
		--env HOME=/work/bin \
		--publish 3000:3000 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c "npx --yes serve $(npx_timeout_command)"

check: bin/check

clean:
	rm -rf bin/

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/check:
	$(MAKE) $(css_target) $(html_target) $(js_target)

bin/check-css: $(css_file_name) .dockerignore .gitignore bin bin/.stylelintrc.json Dockerfile
	docker container run \
		$(interactive_tty_arg) \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		npx --yes js-beautify --end-with-newline --indent-with-tabs --newline-between-rules --no-preserve-newlines --replace --type css $(css_file_name) && \
		cd bin/ && \
		npm install stylelint-config-standard && \
		npm exec stylelint --fix ../$(css_file_name) && \
		cd .. && \
		npx --yes css-validator --profile css3svg $(css_file_name)'
	touch bin/check-css

bin/check-html: $(html_file_name) .dockerignore .gitignore bin Dockerfile
	docker container run \
		$(interactive_tty_arg) \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		npx --yes js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace $(html_file_name) && \
		npx --yes html-validate $(html_file_name)'
	touch bin/check-html

bin/check-js: $(js_file_name) .dockerignore .gitignore bin bin/eslintrc.js Dockerfile
	docker container run \
		$(interactive_tty_arg) \
		--env HOME=/work/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/work/ \
		--workdir /work/ \
		$$(docker image build --quiet .) /bin/sh -c '\
		npx --yes eslint --config bin/eslintrc.js --fix $(js_file_name) && \
		npx --yes js-beautify --end-with-newline --indent-with-tabs --no-preserve-newlines --type js --replace $(js_file_name)'
	touch bin/check-js

bin/eslintrc.js: bin
	printf 'module.exports = { "env": { "browser": true, "es2021": true }, "extends": "eslint:recommended", "overrides": [ ], "parserOptions": { "ecmaVersion": "latest" }, "rules": { "indent": [ "error", "tab" ], "linebreak-style": [ "error", "unix" ], "quotes": [ "error", "single" ], "semi": [ "error", "always" ], "no-undef": 0 } }\n' > bin/eslintrc.js

bin/.stylelintrc.json: bin
	printf '{ "extends": "stylelint-config-standard", "rules": { "indentation": "tab" } }\n' > bin/.stylelintrc.json

Dockerfile:
	printf 'FROM node\n' > Dockerfile
