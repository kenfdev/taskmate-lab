import { setTimeout as delay } from "node:timers/promises";
import {
  DomainError, DomainRepository, digest, type Notification, type Preferences, type Task, type UserId,
} from "./domain.js";

export interface ReminderPort {
  reminder(user: UserId, preferences: Preferences, tasks: Task[]): Promise<string>;
}
export function localDayAndHour(now: Date, timezone: string): { day: string; hour: number } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", hourCycle: "h23",
  }).formatToParts(now);
  const value = (type: Intl.DateTimeFormatPartTypes): string => parts.find((part) => part.type === type)!.value;
  return { day: value("year") + "-" + value("month") + "-" + value("day"), hour: Number(value("hour")) };
}
export class NotificationService {
  constructor(private readonly repo: DomainRepository, private readonly chat: ReminderPort) {}
  async preview(user: UserId): Promise<Notification> {
    return this.repo.withLock("preview:" + user, async () => {
      const preferences = await this.repo.getPreferences(user);
      const tasks = (await this.repo.listTasks(user)).filter((task) => !task.completed);
      const key = "preview:" + digest(JSON.stringify({
        preferences, tasks: tasks.map((task) => ({ id: task.id, title: task.title })),
      }));
      const saved = await this.repo.notification(user, key);
      if (saved) return saved;
      const content = await this.chat.reminder(user, preferences, tasks);
      return this.repo.addNotification(user, key, "preview", content);
    });
  }
  async tick(now = new Date()): Promise<void> {
    for (const user of ["alice", "bob"] as const) {
      try {
        await this.repo.withLock("daily:" + user, async () => {
          const preferences = await this.repo.getPreferences(user);
          const { day, hour } = localDayAndHour(now, preferences.timezone);
          // Catch up after a restart; repeated/DST hours share one local-date key.
          if (hour < preferences.notification_hour) return;
          const key = "daily:" + day;
          if (await this.repo.notification(user, key)) return;
          const tasks = (await this.repo.listTasks(user)).filter((task) => !task.completed);
          if (!tasks.length) return;
          const content = await this.chat.reminder(user, preferences, tasks);
          await this.repo.addNotification(user, key, "daily", content);
        });
      } catch (error) {
        if (error instanceof DomainError && (error.status === 401 || error.status === 409)) continue;
        // A profile failure must not starve the other profile. Never log provider
        // bodies, tokens, request URLs or user task content.
        console.error("Notification generation failed for a demo profile. Check model configuration and connection.");
      }
    }
  }
}
export async function runWorker(
  services: { repo: DomainRepository; chat: ReminderPort },
  signal: AbortSignal,
  intervalMs = 60_000,
): Promise<void> {
  const notifications = new NotificationService(services.repo, services.chat);
  while (!signal.aborted) {
    await notifications.tick();
    try { await delay(intervalMs, undefined, { signal }); }
    catch (error) { if (!signal.aborted) throw error; }
  }
}

async function main(): Promise<void> {
  const { buildServices } = await import("./app.js");
  const services = await buildServices();
  const controller = new AbortController();
  const stop = () => controller.abort();
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  try {
    if (!services.settings.model) throw new DomainError(503, "Set TASKMATE_MODEL after connecting ChatGPT and listing available models, then restart the worker.");
    await runWorker(services, controller.signal);
  } finally {
    process.off("SIGINT", stop);
    process.off("SIGTERM", stop);
    await services.close();
  }
}
const { pathToFileURL } = await import("node:url");
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error(error instanceof DomainError ? error.message : "Worker could not start. Check local configuration and PostgreSQL.");
    process.exitCode = 1;
  });
}
