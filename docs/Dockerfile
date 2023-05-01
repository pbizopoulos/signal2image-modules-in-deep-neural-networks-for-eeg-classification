FROM node:20.0.0
RUN apt-get update && apt-get install -y jq
COPY package.json /usr/src/
RUN npm install --omit=dev --prefix /usr/src
RUN npm install --global $(jq --raw-output ".devDependencies | to_entries[] | \"\(.key)@\(.value)\"" /usr/src/package.json)
