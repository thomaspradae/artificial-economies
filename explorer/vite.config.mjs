import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.GITHUB_PAGES_BASE ?? "/artificial-economies/",
  plugins: [react()],
  esbuild: {
    tsconfigRaw: {
      compilerOptions: {
        jsx: "automatic",
        useDefineForClassFields: true
      }
    }
  },
  build: {
    outDir: "dist",
    target: "es2022",
    sourcemap: true
  }
});
