FROM node:19.2.0
RUN apt-get update && apt-get install -y jq
COPY package.json .
RUN jq -r ".devDependencies | to_entries | map_values( .key + \"@\" + .value ) | join(\"\n\")" package.json | xargs -n 1 npm install --global
