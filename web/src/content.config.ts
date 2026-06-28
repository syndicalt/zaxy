import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";

// Docs are synced from the repo `docs/` tree by scripts/sync-content.mjs.
// generateId preserves the verbatim filename stem (dots + case) so the URL
// scheme matches the legacy build exactly: e.g. announcements/zaxy-v3.0-x-article
// and research/artifacts/ann-2026-06/BASELINE — not slugified variants.
const docs = defineCollection({
  loader: glob({
    pattern: "**/*.{md,mdx}",
    base: "./src/content/docs",
    generateId: ({ entry }) => entry.replace(/\.(md|mdx)$/i, ""),
  }),
});

export const collections = { docs };
