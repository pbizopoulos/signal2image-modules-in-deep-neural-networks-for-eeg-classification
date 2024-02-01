FROM node:21.6.1
RUN apt-get update && apt-get install -y jq
WORKDIR /usr/src/app
COPY package.json /usr/src/
RUN npm install --omit=dev --prefix /usr/src/
RUN npm install --global $(jq --raw-output ".devDependencies | to_entries[] | \"\(.key)@\(.value)\"" /usr/src/package.json)
