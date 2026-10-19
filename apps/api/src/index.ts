import { serve } from "@hono/node-server";
import { fileURLToPath } from "node:url";
import { buildServices, createApp } from "./app.js";

const services = await buildServices();
const staticRoot = fileURLToPath(new URL("../../web/dist", import.meta.url));
const app = createApp(services, { staticRoot });
const server = serve({ fetch: app.fetch, hostname: "127.0.0.1", port: 8765 });
console.info("Taskmate Lab: http://127.0.0.1:8765");

let stopping = false;
async function shutdown() {
  if (stopping) return;
  stopping = true;
  server.close(async () => {
    await services.close();
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 10_000).unref();
}
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);
