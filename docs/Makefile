.POSIX:

.PHONY: all check clean help

DEBUG = 1

container_engine = docker
css_file_name = style.css
css_target = $$(test -s $(css_file_name) && printf '%s' 'bin/check-css')
debug_args = $$(test -t 0 && printf '%s' '--interactive --tty')
html_file_name = index.html
html_target = $$(test -s $(html_file_name) && printf '%s' 'bin/check-html')
js_file_name = script.js
js_target = $$(test -s $(js_file_name) && printf '%s' 'bin/check-js')
npx_timeout_command = $$(test $(DEBUG) = 1 && printf '%s' '& sleep 1; kill $$!')
user_arg = $$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir = /work

all: .dockerignore .gitignore bin/cert.pem
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--publish 8080:8080 \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node /bin/bash -c "npx --yes http-server --cert bin/cert.pem --key bin/key.pem --ssl $(npx_timeout_command)"

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
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		alpine/openssl req -newkey rsa:2048 -subj "/C=../ST=../L=.../O=.../OU=.../CN=.../emailAddress=..." -new -nodes -x509 -days 3650 -keyout bin/key.pem -out bin/cert.pem

bin/check: .dockerignore .gitignore bin
	$(MAKE) $(css_target) $(html_target) $(js_target)

bin/check-css: .dockerignore .gitignore bin $(css_file_name)
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes css-validator --profile css3svg $(css_file_name)
	touch bin/check-css

bin/check-html: $(html_file_name) .dockerignore .gitignore bin
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes html-validate $(html_file_name)
	touch bin/check-html

bin/check-js: $(js_file_name) .dockerignore .gitignore bin bin/eslintrc.js
	$(container_engine) container run \
		$(debug_args) \
		$(user_arg) \
		--env HOME=$(work_dir)/bin \
		--env NODE_PATH=$(work_dir)/bin \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		node npx --yes eslint --fix --config bin/eslintrc.js $(js_file_name)
	touch bin/check-js

bin/eslintrc.js: bin
	echo 'module.exports = { "env": { "browser": true, "es2021": true }, "extends": "eslint:recommended", "overrides": [ ], "parserOptions": { "ecmaVersion": "latest" }, "rules": { "indent": [ "error", "tab" ], "linebreak-style": [ "error", "unix" ], "quotes": [ "error", "single" ], "semi": [ "error", "always" ], "no-undef": 0 } }' > bin/eslintrc.js
