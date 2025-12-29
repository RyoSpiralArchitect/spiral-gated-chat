import type { GateState, TokenLogprob } from "@/lib/types";

export const DIMS = ["RISK", "NOVELTY", "GOAL", "UNCERTAINTY", "OPPORTUNITY", "META"] as const;
export type Dim = (typeof DIMS)[number];

const EPS = 1e-8;

export function createInitialGateState(): GateState {
  return {
    S: { mean: 2.0, var: 1.0 },
    H: { mean: 1.0, var: 1.0 },
    last_state: 0.3,

    // META-cap governor (start conservative)
    meta_cap_stage: 0,
    meta_cap_last_change_turn: -999,

    last_dims: [],
    last_focus: [],
    last_states: [],
    last_pulse_turn: -999,
  };
}

export function emaUpdate(stats: { mean: number; var: number }, x: number, eta: number): { mean: number; var: number } {
  // Exponential moving mean/variance (cheap and stable)
  const mean = (1 - eta) * stats.mean + eta * x;
  const diff = x - mean;
  const var_ = (1 - eta) * stats.var + eta * diff * diff;
  return { mean, var: Math.max(var_, EPS) };
}

export function zScore(stats: { mean: number; var: number }, x: number): number {
  return (x - stats.mean) / Math.sqrt(stats.var + EPS);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function sliceFirstLine(tokens: TokenLogprob[], maxTokens = 30): TokenLogprob[] {
  const out: TokenLogprob[] = [];
  for (const t of tokens) {
    out.push(t);
    if (out.length >= maxTokens) break;
    if (t.token.includes("\n")) break;
  }
  return out;
}

export function meanSurprisal(tokens: TokenLogprob[]): number | null {
  if (!tokens.length) return null;
  const vals = tokens.map((t) => -t.logprob);
  const sum = vals.reduce((a, b) => a + b, 0);
  return sum / vals.length;
}

export function approxEntropy(tokens: TokenLogprob[]): number | null {
  // Approx entropy from top_logprobs. If missing, return null.
  const Hs: number[] = [];
  for (const t of tokens) {
    const top = t.top_logprobs;
    if (!top || top.length === 0) continue;
    // convert to probabilities
    const ps = top.map((x) => Math.exp(x.logprob));
    const sumP = ps.reduce((a, b) => a + b, 0);
    const r = Math.max(0, 1 - sumP);
    let H = 0;
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      if (p > 0) H -= p * Math.log(p + EPS);
    }
    if (r > 0) H -= r * Math.log(r + EPS);
    Hs.push(H);
  }
  if (Hs.length === 0) return null;
  const sum = Hs.reduce((a, b) => a + b, 0);
  return sum / Hs.length;
}

export function hysteresisUpdate(prev: number, raw: number, up = 0.62, down = 0.48, inertia = 0.6): number {
  // Stickiness band: if raw within (down, up), keep prev. Otherwise move toward raw.
  let target = prev;
  if (raw >= up || raw <= down) target = raw;
  const next = inertia * prev + (1 - inertia) * target;
  return clamp01(next);
}

export function parseDimFocus(probeText: string): { dim: string | null; focus: string | null } {
  // Very forgiving parser. Assumes lines like "DIM: RISK" and "FOCUS: xxx".
  const dimMatch = probeText.match(/\bDIM\s*:\s*([A-Z_]+)/i);
  const focusMatch = probeText.match(/\bFOCUS\s*:\s*(.+)/i);
  const dim = dimMatch ? dimMatch[1].toUpperCase() : null;
  const focus = focusMatch ? focusMatch[1].trim() : null;
  return { dim, focus };
}

export function parseProbeFields(probeText: string): {
  dim: string | null;
  focus: string | null;
  next: string | null;
  why: string | null;
} {
  const dimMatch = probeText.match(/\bDIM\s*:\s*([A-Z_]+)/i);
  const focusMatch = probeText.match(/\bFOCUS\s*:\s*(.+)/i);
  const nextMatch = probeText.match(/\bNEXT\s*:\s*(.+)/i);
  const whyMatch = probeText.match(/\bWHY\s*:\s*(.+)/i);
  const dim = dimMatch ? dimMatch[1].toUpperCase() : null;
  const focus = focusMatch ? focusMatch[1].trim() : null;
  const next = nextMatch ? nextMatch[1].trim() : null;
  const why = whyMatch ? whyMatch[1].trim() : null;
  return { dim, focus, next, why };
}

export function updateStagnationBuffers(gs: GateState, dim: string | null, focus: string | null, state: number, maxLen = 12): void {
  if (dim) gs.last_dims.push(dim);
  if (focus) gs.last_focus.push(focus);
  gs.last_states.push(state);
  while (gs.last_dims.length > maxLen) gs.last_dims.shift();
  while (gs.last_focus.length > maxLen) gs.last_focus.shift();
  while (gs.last_states.length > maxLen) gs.last_states.shift();
}

export function variance(xs: number[]): number {
  if (xs.length <= 1) return 0;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((acc, x) => acc + (x - mean) ** 2, 0) / (xs.length - 1);
  return v;
}

export function lowDiversity<T>(xs: T[], uniqMax: number): boolean {
  const s = new Set(xs);
  return s.size <= uniqMax;
}

export function shouldExplorationPulse(gs: GateState): boolean {
  // Conservative: only when state is flat AND DIM is repeating.
  if (gs.last_states.length < 8) return false;
  if (gs.last_dims.length < 8) return false;
  if (gs.last_focus.length < 8) return false;
  const v = variance(gs.last_states.slice(-8));
  const flat = v < 0.002;
  const dimFlat = lowDiversity(gs.last_dims.slice(-8), 1);
  const focusLowDiv = lowDiversity(gs.last_focus.slice(-8), 2);
  return flat && dimFlat && focusLowDiv;
}
