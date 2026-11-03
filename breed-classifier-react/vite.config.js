import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
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
});
