.POSIX:

.PHONY: all check clean help

DEBUG = 1

css_file_name = style.css
css_target = $$(test -s $(css_file_name) && printf 'bin/check-css')
debug_args = $$(test -t 0 && printf '%s' '--interactive --tty')
html_file_name = index.html
html_target = $$(test -s $(html_file_name) && printf 'bin/check-html')
js_file_name = script.js
js_target = $$(test -s $(js_file_name) && printf 'bin/check-js')
npx_timeout_command = $$(test $(DEBUG) = 1 && printf '& sleep 1; kill $$!')
work_dir = /work

all: .dockerignore .gitignore bin/cert.pem
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--publish 8080:8080 \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/sh -c "npx --yes http-server --cert bin/cert.pem --key bin/key.pem --ssl $(npx_timeout_command)"

check: bin/check

clean:
	rm -rf bin/

help:
	@printf 'make all	# Run server (DEBUG=0 for disabling debug).\n'
	@printf 'make check	# Check code.\n'
	@printf 'make clean	# Remove binaries.\n'
	@printf 'make help	# Show help.\n'

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/cert.pem: bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		alpine/openssl req -newkey rsa:2048 -subj "/C=../ST=../L=.../O=.../OU=.../CN=.../emailAddress=..." -new -nodes -x509 -days 3650 -keyout bin/key.pem -out bin/cert.pem

bin/check: .dockerignore .gitignore bin
	$(MAKE) $(css_target) $(html_target) $(js_target)

bin/check-css: .dockerignore .gitignore bin $(css_file_name)
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes css-validator --profile css3svg $(css_file_name)
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes js-beautify --end-with-newline --indent-with-tabs --newline-between-rules --no-preserve-newlines --replace --type css $(css_file_name)
	touch bin/check-css

bin/check-html: $(html_file_name) .dockerignore .gitignore bin
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes html-validate $(html_file_name)
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes js-beautify --end-with-newline --indent-inner-html --indent-with-tabs --no-preserve-newlines --type html --replace $(html_file_name)
	touch bin/check-html

bin/check-js: $(js_file_name) .dockerignore .gitignore bin bin/eslintrc.js
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes eslint --config bin/eslintrc.js --fix $(js_file_name)
	docker container run \
		$(debug_args) \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes js-beautify --end-with-newline --indent-with-tabs --no-preserve-newlines --type js --replace $(js_file_name)
	touch bin/check-js

bin/eslintrc.js: bin
	echo 'module.exports = { "env": { "browser": true, "es2021": true }, "extends": "eslint:recommended", "overrides": [ ], "parserOptions": { "ecmaVersion": "latest" }, "rules": { "indent": [ "error", "tab" ], "linebreak-style": [ "error", "unix" ], "quotes": [ "error", "single" ], "semi": [ "error", "always" ], "no-undef": 0 } }' > bin/eslintrc.js
