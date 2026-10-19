import { defineConfig } from "vite-plus";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    fileParallelism: false,
    testTimeout: 30_000,
    hookTimeout: 30_000,
    reporters: ["default", "junit"],
    outputFile: { junit: "../../artifacts/api-tests.xml" },
  },
});
