#!/usr/bin/env node
// Prebuild: mirror the repo docs tree + public assets into the Astro project.
// Generated content is gitignored — the repo `docs/` remains the source of truth.
import { cpSync, rmSync, existsSync, statSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(webRoot, "..");

const docsSrc = path.join(repoRoot, "docs");
const docsContent = path.join(webRoot, "src/content/docs");
const docsPublic = path.join(webRoot, "public/docs");
const assetsSrc = path.join(repoRoot, "docs/assets");
const assetsDest = path.join(webRoot, "public/assets");
const reportsSrc = path.join(repoRoot, "reports");
const reportsDest = path.join(webRoot, "public/reports");

const isDir = (p) => {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
};
const isMd = (p) => /\.mdx?$/i.test(p);

function fresh(dest) {
  rmSync(dest, { recursive: true, force: true });
  mkdirSync(dest, { recursive: true });
}

// 1) markdown -> content collection (never copy the assets dir into content)
fresh(docsContent);
cpSync(docsSrc, docsContent, {
  recursive: true,
  filter: (src) =>
    !src.split(path.sep).includes("assets") && (isDir(src) || isMd(src)),
});

// 2) every NON-markdown doc file (images, gif, mp4, png frames, assets) ->
//    /docs mirror so relative refs from rendered docs resolve.
fresh(docsPublic);
cpSync(docsSrc, docsPublic, {
  recursive: true,
  filter: (src) => isDir(src) || !isMd(src),
});

// 3) doc assets also mounted at root /assets (homepage og + ../assets refs)
if (existsSync(assetsSrc)) {
  fresh(assetsDest);
  cpSync(assetsSrc, assetsDest, { recursive: true });
}

// 4) reports -> /reports (parity with the legacy Pages deploy)
if (existsSync(reportsSrc)) {
  fresh(reportsDest);
  cpSync(reportsSrc, reportsDest, { recursive: true });
}

console.log("[sync-content] docs (content + /docs mirror) + assets + reports synced");
