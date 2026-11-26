import http from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const apiOrigin = { hostname: "127.0.0.1", port: 8000 };
const port = 5173;
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function proxy(request, response) {
  const upstream = http.request(
    {
      ...apiOrigin,
      method: request.method,
      path: request.url,
      headers: { ...request.headers, host: "127.0.0.1:8000" },
    },
    (result) => {
      response.writeHead(result.statusCode ?? 502, result.headers);
      result.pipe(response);
    },
  );
  upstream.setTimeout(35_000, () => upstream.destroy(new Error("upstream timeout")));
  upstream.on("error", () => {
    if (response.headersSent) {
      response.end();
      return;
    }
    response.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
    response.end(JSON.stringify({ detail: "服务未连接" }));
  });
  request.pipe(upstream);
}

async function serveStatic(pathname, response) {
  try {
    const relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname).replace(/^\/+/, "");
    const filePath = resolve(root, relative);
    if (!filePath.startsWith(`${root}${sep}`)) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    const content = await readFile(filePath);
    response.writeHead(200, {
      "Content-Type": types[extname(filePath)] ?? "application/octet-stream",
      "Cache-Control": "no-store",
    });
    response.end(content);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not Found");
  }
}

http
  .createServer((request, response) => {
    const pathname = new URL(request.url, "http://127.0.0.1").pathname;
    if (pathname === "/health" || pathname === "/ready" || pathname.startsWith("/v1/")) {
      proxy(request, response);
      return;
    }
    void serveStatic(pathname, response);
  })
  .listen(port, "127.0.0.1", () => {
    console.log(`前端：http://127.0.0.1:${port}`);
    console.log("API：http://127.0.0.1:8000");
  });
