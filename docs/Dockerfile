FROM node:19.7.0
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y jq
COPY package.json .
RUN npm install --global $(jq --raw-output ".devDependencies | to_entries | map_values( .key + \"@\" + .value ) | join(\" \")" package.json)
