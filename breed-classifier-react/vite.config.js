import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Copy the ONNX Runtime Web WASM runtime into dist/ort (and keep public/ort in
 * sync for `vite dev`). ONNX Runtime Web loads these at runtime; Vite does not
 * bundle them automatically for the extern-WASM build.
 */
function copyOrtWasm() {
  const sourceDir = path.resolve(__dirname, "node_modules/onnxruntime-web/dist");
  const publicDir = path.resolve(__dirname, "public/ort");
  const distDir = path.resolve(__dirname, "dist/ort");
  const files = [
    "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd-threaded.mjs",
  ];
  const copy = (targetDir) => {
    fs.mkdirSync(targetDir, { recursive: true });
    for (const file of files) {
      const source = path.join(sourceDir, file);
      if (fs.existsSync(source)) fs.copyFileSync(source, path.join(targetDir, file));
    }
  };
  return {
    name: "copy-ort-wasm",
    buildStart() {
      copy(publicDir);
    },
    closeBundle() {
      copy(distDir);
    },
  };
}

export default defineConfig(({ command }) => {
  const isServe = command === "serve";
  return {
    plugins: [react(), copyOrtWasm()],
    // ONNX Runtime Web ships pre-minified ESM + WASM; excluding it from
    // dependency pre-bundling keeps its dynamic WASM loader intact.
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
    resolve: {
      alias: isServe
        ? {}
        : {
            // Production/Capacitor: extern-WASM build + public/ort files.
            // Dev uses the bundled build because Vite cannot transform a
            // dynamically imported .mjs from public/.
            "onnxruntime-web/wasm": path.resolve(
              __dirname,
              "node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
            ),
          },
    },
    build: {
      target: "es2020",
      sourcemap: false,
    },
    server: {
      port: 5173,
      watch: {
        ignored: ["**/android/**"],
      },
      proxy: {
        // Forward all API calls to the Flask backend
        "/api": {
          target: "http://localhost:5000",
          changeOrigin: true,
        },
        "/socket.io": {
          target: "http://localhost:5000",
          changeOrigin: true,
          ws: true,
          configure: (proxy, _options) => {
            proxy.on("error", (err, _req, _res) => {
              // Ignore transient disconnect / refresh noise from socket.io
              const code = err.code || "";
              if (["ECONNRESET", "ECONNABORTED", "EPIPE"].includes(code)
                  || /ECONNRESET|ECONNABORTED|EPIPE/.test(err.message || "")) {
                return;
              }
              console.error("[vite socket proxy]", err.message);
            });
          },
        },
      },
    },
  };
});
