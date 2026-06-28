import path from "node:path";
import { existsSync } from "node:fs";
import { visit } from "unist-util-visit";

// Rewrite repo-relative Markdown links/images so rendered docs resolve on the
// static site, preserving the legacy URL scheme:
//   foo.md                         -> /docs/<resolved>.html
//   ../reports/x/report.md|json    -> /reports/x/report.md|json   (self-hosted)
//   ../assets/x.png                -> /assets/x.png
//   ../../ZAXY-3.md, ../scripts/x  -> GitHub blob link
//   relative image (media/x.gif)   -> left relative (mirrored under /docs/)
// Absolute, anchor, http(s), mailto, data, tel links are untouched.

const GH_BLOB = "https://github.com/syndicalt/zaxy/blob/master";

function splitHash(u) {
  const i = u.indexOf("#");
  return i === -1 ? [u, ""] : [u.slice(0, i), u.slice(i)];
}

export function remarkZaxyLinks() {
  return (tree, file) => {
    const cwd = process.cwd();
    const docsRoot = path.resolve(cwd, "src/content/docs");
    const publicRoot = path.resolve(cwd, "public");
    const srcPath =
      file.path || (file.history && file.history[file.history.length - 1]) || "";
    const dir = srcPath ? path.dirname(srcPath) : docsRoot;

    // repo-relative path of a link target (docs/ corresponds to docsRoot)
    const repoPathOf = (p) => {
      const resolved = path.resolve(dir, p);
      const rel = path.relative(docsRoot, resolved).split(path.sep).join("/");
      return { repoPath: path.posix.normalize("docs/" + rel), rel };
    };

    const rewriteLink = (url) => {
      if (!url || /^(https?:|mailto:|data:|tel:|#|\/)/i.test(url)) return url;
      const [p, hash] = splitHash(url);
      if (!p) return url;
      const { repoPath, rel } = repoPathOf(p);

      // markdown doc inside docs/
      if (
        (p.endsWith(".md") || p.endsWith(".mdx")) &&
        repoPath.startsWith("docs/") &&
        !rel.startsWith("..")
      ) {
        return "/docs/" + rel.replace(/\.(md|mdx)$/i, "") + ".html" + hash;
      }
      // reports/ — self-hosted if present, else GitHub
      if (repoPath.startsWith("reports/")) {
        return existsSync(path.join(publicRoot, repoPath))
          ? "/" + repoPath + hash
          : `${GH_BLOB}/${repoPath}${hash}`;
      }
      // doc assets mounted at /assets
      if (repoPath.startsWith("docs/assets/")) {
        return "/" + repoPath.slice("docs/".length) + hash;
      }
      // any other repo file (ZAXY-3.md, AGENTS.md, scripts/…) -> GitHub
      return `${GH_BLOB}/${repoPath}${hash}`;
    };

    const rewriteImage = (url) => {
      if (!url || /^(https?:|data:|\/)/i.test(url)) return url;
      const [p, hash] = splitHash(url);
      const { repoPath } = repoPathOf(p);
      if (repoPath.startsWith("docs/assets/")) {
        return "/" + repoPath.slice("docs/".length) + hash;
      }
      // leave other relative images alone; they resolve under the mirrored
      // /docs/<dir>/ tree copied into public/docs by sync-content.mjs.
      return url;
    };

    visit(tree, "link", (n) => {
      n.url = rewriteLink(n.url);
    });
    visit(tree, "definition", (n) => {
      n.url = rewriteLink(n.url);
    });
    visit(tree, "image", (n) => {
      n.url = rewriteImage(n.url);
    });
  };
}

export default remarkZaxyLinks;
