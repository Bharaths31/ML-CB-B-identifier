import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");
const MAIN = path.resolve(REPO, "breed-classifier-react");

/**
 * Standalone Veterinary Review Portal.
 *
 * It shares the main app's source modules (API client, socket hook, i18n,
 * theme switcher, stylesheet) but runs as its own Vite app on port 5174, so the
 * farmer-facing app no longer carries the expert workflow.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    fs: { allow: [REPO] },
    proxy: {
      "/api": "http://localhost:5000",
      "/socket.io": {
        target: "http://localhost:5000",
        ws: true,
        configure: (proxy) => {
          proxy.on("error", (err) => {
            const code = err.code || "";
            if (["ECONNRESET", "ECONNABORTED", "EPIPE"].includes(code)
                || /ECONNRESET|ECONNABORTED|EPIPE/.test(err.message || "")) {
              return;
            }
            console.error("[vet socket proxy]", err.message);
          });
        },
      },
    },
  },
  resolve: {
    dedupe: ["react", "react-dom"],
    alias: {
      react: path.join(MAIN, "node_modules", "react"),
      "react-dom": path.join(MAIN, "node_modules", "react-dom"),
    },
  },
});
