export type TopTokenLogprob = {
  token: string;
  logprob: number;
  bytes?: number[];
};

export type TokenLogprob = {
  token: string;
  logprob: number;
  bytes?: number[];
  top_logprobs?: TopTokenLogprob[];
};

export type ProbeMetrics = {
  surprisal: number | null;
  entropy: number | null;
  score: number | null;
  state: number;
};

export type RunningStats = {
  mean: number;
  var: number;
};

export type GateState = {
  // running baselines for surprisal and entropy (EMA mean/var)
  S: RunningStats;
  H: RunningStats;
  // last state for hysteresis/smoothing
  last_state: number;

  // META-cap governor (staged relaxation)
  // 0: cap 0.55 → 1: cap 0.65 → 2: no cap
  meta_cap_stage: 0 | 1 | 2;
  meta_cap_last_change_turn: number;

  // for stagnation detection
  last_dims: string[];
  last_focus: string[];
  last_states: number[];
  // exploration pulse cooldown
  last_pulse_turn: number;
};

export type MemoryMode = "LOW" | "MID" | "HIGH";

export type AttentionLogEntry = {
  turn: number;
  dim: string;
  focus: string;
  next?: string | null;
};

export type MemoryFragment = {
  id: string;
  key: string;
  turn: number;
  dim: string | null;
  focus: string | null;
  text: string;
  // salience in [0, 1.2] (higher = more likely to be injected)
  salience: number;
  // last turn injected/used (for light rehearsal)
  last_used_turn: number;
};

export type MemoryState = {
  // one-line running summary (used in MID/HIGH modes)
  summary: string;
  summary_updated_turn: number;
  // recent attention frames (used in HIGH mode)
  attn_log: AttentionLogEntry[];
  // salience-ranked short memory fragments ("notes")
  fragments: MemoryFragment[];
};
