import type { MemoryFragment } from "@/lib/types";

function clamp(a: number, b: number, x: number): number {
  return Math.max(a, Math.min(b, x));
}

function clamp01(x: number): number {
  return clamp(0, 1, x);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function makeFragmentId(turn: number): string {
  // good enough for a demo; avoids requiring crypto in edge runtimes
  return `f_${turn}_${Math.random().toString(16).slice(2, 8)}`;
}

export function makeFragmentKey(args: { dim: string | null; focus: string | null; next?: string | null }): string {
  const d = (args.dim ?? "").toUpperCase();
  const f = (args.focus ?? "").trim().toLowerCase();
  const n = (args.next ?? "").trim().toLowerCase();
  return `${d}|${f}|${n}`;
}

export function dimBonus(dim: string | null): number {
  switch ((dim ?? "").toUpperCase()) {
    case "RISK":
      return 0.22;
    case "GOAL":
      return 0.16;
    case "NOVELTY":
      return 0.12;
    case "OPPORTUNITY":
      return 0.10;
    case "UNCERTAINTY":
      return 0.08;
    case "META":
      return 0.04;
    default:
      return 0.0;
  }
}

export function salienceDecayFactor(state: number): number {
  // low state → fast decay; high state → slow decay
  // state=0 => 0.55, state=1 => 0.93
  const t = Math.pow(clamp01(state), 0.9);
  return lerp(0.55, 0.93, t);
}

export function decayFragments(frags: MemoryFragment[], state: number): { factor: number } {
  const factor = salienceDecayFactor(state);
  for (const f of frags) {
    f.salience = clamp(0, 1.2, f.salience * factor);
  }
  return { factor };
}

export function fragmentText(args: { dim: string | null; focus: string | null; next?: string | null }): string {
  const dim = args.dim ? args.dim.toUpperCase() : null;
  const focus = (args.focus ?? "").replace(/\s+/g, " ").trim();
  const next = (args.next ?? "").replace(/\s+/g, " ").trim();
  const head = dim ? `(${dim}) ` : "";
  const body = focus || next ? `${focus}${next ? ` → ${next}` : ""}` : "";
  const t = (head + body).trim();
  return t.length > 160 ? t.slice(0, 160) : t;
}

export function upsertFragment(frags: MemoryFragment[], incoming: Omit<MemoryFragment, "id" | "last_used_turn">, turn: number): {
  added: boolean;
  merged: boolean;
  id: string;
} {
  const key = incoming.key;
  const existing = frags.find((f) => f.key === key);
  if (existing) {
    // merge: keep the newest text, bump salience
    existing.text = incoming.text;
    existing.dim = incoming.dim;
    existing.focus = incoming.focus;
    existing.turn = incoming.turn;
    existing.salience = clamp(0, 1.2, Math.max(existing.salience, incoming.salience) + 0.08);
    return { added: false, merged: true, id: existing.id };
  }
  const id = makeFragmentId(turn);
  frags.push({ ...incoming, id, last_used_turn: -999 });
  return { added: true, merged: false, id };
}

export function pruneFragments(frags: MemoryFragment[], turn: number, maxKeep = 40): void {
  // drop very low salience items that are also old-ish
  const kept = frags.filter((f) => !(f.salience < 0.06 && turn - f.turn > 8));
  kept.sort((a, b) => b.salience - a.salience);
  frags.length = 0;
  for (const f of kept.slice(0, maxKeep)) frags.push(f);
}

export function pickTopFragments(frags: MemoryFragment[], k: number): MemoryFragment[] {
  if (k <= 0) return [];
  const sorted = [...frags].sort((a, b) => b.salience - a.salience);
  return sorted.slice(0, k);
}

export function rehearseFragments(frags: MemoryFragment[], picked: MemoryFragment[], turn: number): void {
  const pickedIds = new Set(picked.map((p) => p.id));
  for (const f of frags) {
    if (!pickedIds.has(f.id)) continue;
    f.last_used_turn = turn;
    f.salience = clamp(0, 1.2, f.salience + 0.05);
  }
}

export function formatFragmentsForPrompt(picked: MemoryFragment[]): string {
  if (!picked.length) return "";
  const lines = picked.map((f) => `- ${f.text}`);
  return [
    "MEMORY_FRAGMENTS (salience-ranked; use ONLY if relevant; ignore if not needed):",
    ...lines,
  ].join("\n");
}
