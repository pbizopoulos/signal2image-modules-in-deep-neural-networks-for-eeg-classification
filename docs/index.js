"use strict";
const fs = require("fs");
const https = require("https");
const serveHandler = require("serve-handler");
const server = https.createServer(
	{
		cert: fs.readFileSync("tmp/fullchain.pem"),
		key: fs.readFileSync("tmp/privkey.pem"),
	},
	(request, response) => {
		return serveHandler(request, response);
	},
);
if (process.env.DEBUG !== "1") {
	server.listen(8000, "172.17.0.2", () => {
		console.log("Running at https://172.17.0.2:8000");
		process.on("SIGINT", () => {
			process.exit(0);
		});
	});
}
