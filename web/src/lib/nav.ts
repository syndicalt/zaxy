// Sidebar model for the docs. Titles come from each doc's first H1; ordering
// is curated for the front docs, then grouped by directory for the long tail.

export interface DocEntry {
  id: string;
  body?: string;
}
export interface NavItem {
  slug: string;
  href: string;
  title: string;
}
export interface NavGroup {
  label: string;
  items: NavItem[];
}

const SECTIONS: { label: string; slugs: string[] }[] = [
  {
    label: "Start here",
    slugs: [
      "getting-started",
      "why-zaxy",
      "mcp-quickstart",
      "first-run-validation",
      "coordinate-quickstart",
      "workspace-genesis",
    ],
  },
  {
    label: "Guides",
    slugs: [
      "architecture",
      "configuration",
      "mcp",
      "mcp-install-targets",
      "integrations",
      "embeddings",
      "retrieval",
      "consolidation",
      "crystallization",
      "editability",
      "agent-events",
      "hooks",
      "packet-analyzer",
      "codebase",
      "external-ingest",
      "plugins",
      "coordinate-roadmap",
    ],
  },
  {
    label: "Reference",
    slugs: [
      "api",
      "api-inventory",
      "graph-schema",
      "eventloom",
      "export-contract",
      "security",
      "stability-commitment",
    ],
  },
  {
    label: "Operations & evidence",
    slugs: [
      "deployment",
      "operations",
      "runbook",
      "testing",
      "migration",
      "benchmarks",
      "external-validation",
    ],
  },
];

const DIR_GROUPS: { dir: string; label: string }[] = [
  { dir: "announcements", label: "Announcements" },
  { dir: "research", label: "Research" },
  { dir: "superpowers/plans", label: "Roadmaps" },
  { dir: "superpowers/specs", label: "Specs" },
  { dir: "essays", label: "Essays" },
  { dir: "experimental", label: "Experimental" },
  { dir: "media", label: "Media" },
  { dir: "archive", label: "Archive" },
];

export function titleFor(entry: DocEntry): string {
  const body = entry.body ?? "";
  for (const raw of body.split("\n")) {
    const m = raw.match(/^#\s+(.+?)\s*$/);
    if (m) return m[1].replace(/[*`_]/g, "").trim();
  }
  const base = entry.id.split("/").pop() ?? entry.id;
  return base.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const href = (slug: string) => `/docs/${slug}.html`;

export function buildDocsNav(entries: DocEntry[]): NavGroup[] {
  const byId = new Map(entries.map((e) => [e.id, e]));
  const used = new Set<string>();
  const groups: NavGroup[] = [];

  for (const sec of SECTIONS) {
    const items: NavItem[] = [];
    for (const slug of sec.slugs) {
      const e = byId.get(slug);
      if (!e) continue;
      used.add(slug);
      items.push({ slug, href: href(slug), title: titleFor(e) });
    }
    if (items.length) groups.push({ label: sec.label, items });
  }

  for (const g of DIR_GROUPS) {
    const prefix = g.dir + "/";
    const items = entries
      .filter((e) => e.id.startsWith(prefix) && !used.has(e.id))
      .sort((a, b) => b.id.localeCompare(a.id))
      .map((e) => {
        used.add(e.id);
        return { slug: e.id, href: href(e.id), title: titleFor(e) };
      });
    if (items.length) groups.push({ label: g.label, items });
  }

  // any unplaced docs
  const rest = entries
    .filter((e) => !used.has(e.id))
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((e) => ({ slug: e.id, href: href(e.id), title: titleFor(e) }));
  if (rest.length) groups.push({ label: "More", items: rest });

  return groups;
}
