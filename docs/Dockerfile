FROM node:19.9.0
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y jq
COPY package.json /usr/src/
RUN npm install --omit=dev --prefix /usr/src
RUN npm install --global $(jq --raw-output ".devDependencies | to_entries[] | \"\(.key)@\(.value)\"" /usr/src/package.json)
