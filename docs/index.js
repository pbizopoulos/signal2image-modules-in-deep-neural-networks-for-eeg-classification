"use strict";
const fs = require("fs");
const handler = require("serve-handler");
const https = require("https");

const options = {
	key: fs.readFileSync("tmp/key.pem"),
	cert: fs.readFileSync("tmp/cert.pem"),
};

const server = https.createServer(options, (request, response) => {
	return handler(request, response);
});

if (process.env.DEBUG !== "1") {
	server.listen(8000, "172.17.0.2", () => {
		console.log("Running at https://172.17.0.2:8000");
	});
}
