import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // All /api/* requests → backend at localhost:8000 (strips /api prefix)
      // e.g. /api/reset → http://localhost:8000/reset
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: (proxy) => {
          proxy.on("error", (err) => {
            console.log("[Vite Proxy] Backend unreachable:", err.message);
          });
          proxy.on("proxyReq", (_proxyReq, req) => {
            console.log("[Vite Proxy] →", req.method, req.url);
          });
          proxy.on("proxyRes", (proxyRes, req) => {
            console.log("[Vite Proxy] ←", proxyRes.statusCode, req.url);
          });
        },
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
