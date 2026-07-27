import {defineConfig} from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.browser.test.mjs",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI?1:0,
  timeout: 30_000,
  use:{
    browserName:"chromium",
    headless:true,
  },
  reporter:process.env.CI?"github":"line",
});
