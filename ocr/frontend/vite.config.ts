import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api":       { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/projects":  { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/documents": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/jobs":      { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/files":     { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/system":    { target: "http://127.0.0.1:8000", changeOrigin: true },  // ← new
    },
  },
});