import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      "/api": {
        target: process.env.API_GATEWAY_URL || "https://analysis-be.pathtodeal.com",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  preview: {
    port: 3000,
    host: true,
    allowedHosts: ["analysis.pathtodeal.com"],
    proxy: {
      "/api": {
        target: process.env.API_GATEWAY_URL || "https://analysis-be.pathtodeal.com",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
