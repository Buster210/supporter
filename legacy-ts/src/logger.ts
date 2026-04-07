import * as fs from "node:fs";
import { getFileSink } from "@logtape/file";
import { configure, getLogger } from "@logtape/logtape";
import * as dotenv from "dotenv";

dotenv.config();

export const logger = getLogger(["supporter"]);

/**
 * Initializes the logging system.
 * Configured with error, info, and debug levels.
 * Logs are written to 'app.log' by default.
 */
export async function initLogger() {
  // Truncate the log file to start fresh each run
  try {
    fs.writeFileSync("app.log", "");
  } catch (e) {
    // Ignore if file doesn't exist yet
  }

  const logLevel = (process.env.LOG_LEVEL || "info").toLowerCase();

  await configure({
    sinks: {
      file: getFileSink("app.log", {
        formatter(record) {
          const timestamp = new Date(record.timestamp).toLocaleString();
          const level = record.level.toUpperCase().padEnd(5);
          const category = record.category
            .filter((c) => c !== "supporter")
            .join(".");
          const catStr = category ? ` [${category}]` : "";
          return `${timestamp} [${level}]${catStr} ${record.message}\n`;
        },
      }),
    },
    filters: {},
    loggers: [
      {
        category: ["supporter"],
        level: logLevel as any,
        sinks: ["file"],
      },
      {
        category: ["logtape", "meta"],
        level: "fatal",
        sinks: ["file"],
      },
    ],
  });

  logger.info`Logging initialized at level: ${logLevel}`;
}
