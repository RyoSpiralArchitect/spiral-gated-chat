import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const root = process.cwd();
const port = Number(process.env.E2E_PORT || 3100);
const baseUrl = `http://127.0.0.1:${port}`;
const live = process.env.LIVE_E2E === "1";
const logDir = await mkdtemp(path.join(tmpdir(), "spiral-gated-chat-e2e-"));

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForServer() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(baseUrl);
      if (res.ok) return;
    } catch {
      // keep polling
    }
    await sleep(500);
  }
  throw new Error(`Timed out waiting for ${baseUrl}`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const env = {
  ...process.env,
  PORT: String(port),
  SPIRAL_CHAT_LOG_DIR: logDir,
  ...(live ? {} : { SPIRAL_CHAT_PROVIDER: "mock" }),
};

const child = spawn("npx", ["next", "dev", "-p", String(port), "-H", "127.0.0.1"], {
  cwd: root,
  env,
  stdio: ["ignore", "pipe", "pipe"],
});

let output = "";
child.stdout.on("data", (chunk) => {
  output += chunk.toString();
});
child.stderr.on("data", (chunk) => {
  output += chunk.toString();
});

try {
  await waitForServer();

  const sessionId = `e2e-${Date.now()}`;
  const res = await fetch(`${baseUrl}/api/step`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      sessionId,
      userText: "この設計、どこが死にやすい？状態ゲートのログも見たい。",
    }),
  });

  const body = await res.json();
  assert(res.ok, `POST /api/step failed: ${res.status} ${JSON.stringify(body)}`);
  assert(typeof body.assistantText === "string" && body.assistantText.length > 0, "assistantText missing");
  assert(body.debug?.provider?.name, "provider debug missing");
  assert(body.debug?.provider?.state_source, "state source missing");
  assert(body.debug?.log?.saved === true, "turn log was not saved");

  const logPath = path.join(logDir, "sessions", `${sessionId}.jsonl`);
  const line = (await readFile(logPath, "utf8")).trim();
  const entry = JSON.parse(line);
  assert(entry.schema_version === "spiral-gated-chat.turn.v1", "unexpected log schema version");
  assert(entry.provider === body.debug.provider.name, "log provider mismatch");
  assert(entry.state_source === body.debug.provider.state_source, "log state source mismatch");
  assert(Array.isArray(entry.calls) && entry.calls.length >= 2, "provider calls missing from log");

  const indexPath = path.join(logDir, "session-index.jsonl");
  const sessionIndex = (await readFile(indexPath, "utf8")).trim().split("\n").map((row) => JSON.parse(row));
  assert(sessionIndex.some((row) => row.sessionId === sessionId && row.log_path.endsWith(`${sessionId}.jsonl`)), "session index missing session");

  console.log(
    JSON.stringify(
      {
        ok: true,
        url: baseUrl,
        provider: body.debug.provider.name,
        state_source: body.debug.provider.state_source,
        log: logPath,
        session_index: indexPath,
      },
      null,
      2
    )
  );
} finally {
  child.kill("SIGTERM");
  await sleep(500);
  if (!child.killed) child.kill("SIGKILL");
  if (process.env.KEEP_E2E_LOGS !== "1") {
    await rm(logDir, { recursive: true, force: true });
  }
}

child.on("exit", (code) => {
  if (code && code !== 0 && !output.includes("SIGTERM")) {
    console.error(output);
  }
});
