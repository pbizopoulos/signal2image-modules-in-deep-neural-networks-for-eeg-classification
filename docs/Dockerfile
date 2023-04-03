FROM node:19.8.1
WORKDIR /usr/src
COPY package.json .
RUN npm install --omit=dev
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y jq
COPY package.json .
RUN npm install --global $(jq --raw-output ".devDependencies | to_entries | map_values( .key + \"@\" + .value ) | join(\" \")" package.json)
