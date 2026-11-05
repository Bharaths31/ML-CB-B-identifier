"use strict";

const http = require("node:http");
const advisory = require("../api/advisory");

const port = Number(process.env.PORT || 3001);

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const query = Object.fromEntries(url.searchParams.entries());
  advisory({ method: req.method, query }, res);
});

server.listen(port, () => {
  console.log(`Advisory API listening on http://localhost:${port}/api/advisory`);
});
