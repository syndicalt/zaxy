// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { remarkZaxyLinks } from "./src/remark-zaxy-links.mjs";

// Zaxy 3 site. Static output → GitHub Pages at docs.zaxy.io.
// `file` format preserves the existing public URL scheme exactly:
// docs/<path>.html (1:1 parity with the retired build-site-docs.py output),
// while index routes (homepage + /docs/, /install/) still emit index.html.
export default defineConfig({
  site: "https://docs.zaxy.io",
  trailingSlash: "ignore",
  build: { format: "file" },
  integrations: [mdx(), sitemap()],
  markdown: {
    remarkPlugins: [remarkMath, remarkZaxyLinks],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { theme: "vitesse-dark", wrap: false },
  },
});
