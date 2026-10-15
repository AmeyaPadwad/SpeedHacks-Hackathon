/// <reference types="vitest" />
import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Pette ships as a static web app: `npm run build` writes dist/, which any static host can serve.
// Screen capture, the webcam and the File System Access API need a secure context (https or localhost).
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
    // debug.html is the /debug dashboard as its own page, so static hosts serve it without a rewrite rule.
    rollupOptions: { input: { main: path.resolve(__dirname, "index.html"), debug: path.resolve(__dirname, "debug.html") } },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
