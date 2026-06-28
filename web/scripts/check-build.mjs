#!/usr/bin/env node
// Postbuild gate: fail the build on a broken internal link or a missing
// required doc. Replaces the freshness/link role of the retired
// scripts/build-site-docs.py + tests/test_docs_site.py.
import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(webRoot, "dist");

if (!existsSync(dist)) {
  console.error("[check-build] dist/ not found — run a build first");
  process.exit(1);
}

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

const allFiles = walk(dist);
const htmlFiles = allFiles.filter((f) => f.endsWith(".html"));
const distRel = new Set(allFiles.map((f) => "/" + path.relative(dist, f).split(path.sep).join("/")));

const exists = (urlPath) => {
  let p = urlPath.split("#")[0].split("?")[0];
  if (!p) return true;
  if (p.endsWith("/")) p += "index.html";
  if (distRel.has(p)) return true;
  // directory index fallback
  if (distRel.has(p.replace(/\/?$/, "/index.html"))) return true;
  // on-disk check (covers assets/reports copied verbatim)
  const onDisk = path.join(dist, p.replace(/^\//, ""));
  return existsSync(onDisk) && statSync(onDisk).isFile();
};

const LINK_RE = /(?:href|src)\s*=\s*"([^"]+)"/gi;
const broken = [];

for (const file of htmlFiles) {
  const rel = "/" + path.relative(dist, file).split(path.sep).join("/");
  const baseDir = path.posix.dirname(rel);
  const html = readFileSync(file, "utf8");
  let m;
  while ((m = LINK_RE.exec(html))) {
    const raw = m[1].trim();
    if (
      !raw ||
      raw.startsWith("#") ||
      raw.startsWith("//") ||
      /^(https?:|mailto:|data:|tel:|javascript:)/i.test(raw)
    ) {
      continue;
    }
    const abs = raw.startsWith("/")
      ? raw
      : path.posix.normalize(path.posix.join(baseDir, raw));
    if (!exists(abs)) broken.push({ file: rel, link: raw, abs });
  }
}

const REQUIRED = [
  "/index.html",
  "/docs.html",
  "/install.html",
  "/README.html",
  "/docs/getting-started.html",
  "/docs/why-zaxy.html",
  "/docs/architecture.html",
  "/docs/configuration.html",
  "/docs/mcp.html",
  "/docs/mcp-quickstart.html",
  "/docs/api.html",
  "/docs/api-inventory.html",
  "/docs/eventloom.html",
  "/docs/security.html",
  "/docs/retrieval.html",
  "/docs/integrations.html",
  "/docs/deployment.html",
  "/docs/operations.html",
  "/docs/runbook.html",
  "/docs/benchmarks.html",
  "/docs/external-validation.html",
  "/docs/announcements/zaxy-v3.0-x-article.html",
];
const missing = REQUIRED.filter((p) => !distRel.has(p));

let ok = true;
if (missing.length) {
  ok = false;
  console.error(`[check-build] ${missing.length} required doc(s) missing:`);
  for (const p of missing) console.error("  - " + p);
}
if (broken.length) {
  ok = false;
  console.error(`[check-build] ${broken.length} broken internal link(s):`);
  for (const b of broken.slice(0, 40)) {
    console.error(`  - ${b.file} → ${b.link}`);
  }
  if (broken.length > 40) console.error(`  … +${broken.length - 40} more`);
}

if (!ok) process.exit(1);
console.log(
  `[check-build] OK — ${htmlFiles.length} pages, links resolve, ${REQUIRED.length} required docs present`,
);
