import OpenAI from "openai";
import {
  approxEntropy,
  hysteresisUpdate,
  meanSurprisal,
  parseProbeFields,
  shouldExplorationPulse,
  sliceFirstLine,
  updateStagnationBuffers,
} from "@/lib/gating";
import { getSession } from "@/lib/sessionStore";
import type { AttentionLogEntry, GateState, TokenLogprob } from "@/lib/types";
import {
  explorationSystemPrompt,
  frameSystemPrompt,
  mainSystemPrompt,
  probeSystemPrompt,
  summaryUpdateSystemPrompt,
  verifyPickSystemPrompt,
} from "@/lib/prompts";

import {
  decayFragments,
  dimBonus,
  formatFragmentsForPrompt,
  fragmentText,
  makeFragmentKey,
  pickTopFragments,
  pruneFragments,
  rehearseFragments,
  upsertFragment,
} from "@/lib/fragments";

export const runtime = "nodejs";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(a: number, b: number, x: number): number {
  return Math.max(a, Math.min(b, x));
}

function clamp01(x: number): number {
  return clamp(0, 1, x);
}

function normalizeTokenLogprobs(raw: any): TokenLogprob[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw as TokenLogprob[];
  // common shapes
  if (Array.isArray(raw.content)) return raw.content as TokenLogprob[];
  if (Array.isArray(raw.tokens)) return raw.tokens as TokenLogprob[];
  return [];
}

function extractFirstTextAndLogprobs(response: any): { text: string; logprobs: TokenLogprob[] } {
  const msg = Array.isArray(response?.output)
    ? response.output.find((o: any) => o?.type === "message" && o?.role === "assistant")
    : null;
  const part = msg?.content?.find((c: any) => c?.type === "output_text") ?? msg?.content?.[0];
  const text: string = part?.text ?? "";
  const logprobs = normalizeTokenLogprobs(part?.logprobs);
  return { text, logprobs };
}

type ProbeFields = {
  raw: string;
  dim: string | null;
  focus: string | null;
  next: string | null;
  why: string | null;
};

function safeOneLine(s: string): string {
  return (s ?? "").replace(/\s+/g, " ").trim();
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function powEase(x: number, gamma: number): number {
  return Math.pow(clamp01(x), gamma);
}

function metaCapValue(stage: 0 | 1 | 2): number | null {
  if (stage === 0) return 0.55;
  if (stage === 1) return 0.65;
  return null; // unlocked
}

function recentMetaShare(gs: GateState, window = 12): number | null {
  const dims = gs.last_dims.slice(-window);
  if (dims.length < window) return null;
  const meta = dims.filter((d) => d === "META").length;
  return meta / dims.length;
}

function maybeRelaxMetaCapStage(gs: GateState, turn: number): { changed: boolean; metaShare: number | null } {
  // Monotonic relaxation: 0 (0.55) → 1 (0.65) → 2 (unlocked)
  const share = recentMetaShare(gs, 12);
  if (share === null) return { changed: false, metaShare: null };

  const stage0MinInterval = 12;
  const stage1MinInterval = 18;
  const minInterval = gs.meta_cap_stage === 1 ? stage1MinInterval : stage0MinInterval;
  if (turn - gs.meta_cap_last_change_turn < minInterval) return { changed: false, metaShare: share };

  if (gs.meta_cap_stage === 0 && share <= 0.25) {
    gs.meta_cap_stage = 1;
    gs.meta_cap_last_change_turn = turn;
    return { changed: true, metaShare: share };
  }

  if (gs.meta_cap_stage === 1 && share <= 0.17) {
    gs.meta_cap_stage = 2;
    gs.meta_cap_last_change_turn = turn;
    return { changed: true, metaShare: share };
  }

  return { changed: false, metaShare: share };
}

function applyDimWeight(
  rawState: number,
  dim: string | null,
  gs: GateState
): { raw: number; notes: string[]; metaCap: number | null } {
  let s = rawState;
  const notes: string[] = [];

  // RISK: boost more when we're in low-compute mode (state-dependent)
  if (dim === "RISK") {
    const base = clamp01(s);
    const boost = lerp(0.22, 0.08, base); // low state → bigger boost
    s = clamp01(s + boost);
    notes.push(`DIM=RISK → state +${boost.toFixed(2)} (state-dependent)`);
  }

  // META: staged cap (relaxes over time if META isn't dominating)
  let cap: number | null = null;
  if (dim === "META") {
    cap = metaCapValue(gs.meta_cap_stage);
    if (cap !== null) {
      if (s > cap) notes.push(`DIM=META → state cap ${cap} (stage ${gs.meta_cap_stage})`);
      s = Math.min(s, cap);
    } else {
      notes.push("DIM=META → cap unlocked");
    }
  }

  return { raw: s, notes, metaCap: cap };
}

type PulseInfo = {
  triggered: boolean;
  stagnation_detected: boolean;
  repeating_dim: string | null;
  candidates_text: string | null;
  picked: number | null;
  selected_probe: string | null;
};

function splitCandidateBlocks(text: string): string[] {
  return safeOneLine(text)
    ? text
        .split(/\n\s*\n/g)
        .map((b) => b.trim())
        .filter((b) => b.length > 0)
    : [];
}

function parseCandidates(text: string): ProbeFields[] {
  const blocks = splitCandidateBlocks(text);
  const cands: ProbeFields[] = [];
  for (const b of blocks) {
    const lines = b
      .split("\n")
      .map((x) => x.trimEnd())
      .filter((x) => x.length > 0)
      .slice(0, 4);
    if (lines.length < 2) continue;
    const raw = lines.join("\n");
    const fields = parseProbeFields(raw);
    if (!fields.dim && !fields.focus) continue;
    cands.push({ raw, ...fields });
    if (cands.length >= 3) break;
  }
  return cands;
}

function parsePick(text: string): number | null {
  const m = text.match(/\bPICK\s*:\s*([123])\b/i);
  if (!m) return null;
  return parseInt(m[1], 10);
}

function formatAttnLog(attn: AttentionLogEntry[], maxItems = 8): string {
  const tail = attn.slice(-maxItems);
  const lines = tail.map((e, i) => `${i + 1}. [t${e.turn}] ${e.dim} / ${e.focus}${e.next ? ` → ${e.next}` : ""}`);
  return [
    "ATTENTION_LOG (latest last):",
    ...lines,
    "Use this only as a lightweight memory of what has been salient.",
  ].join("\n");
}

async function updateOneLineSummary(args: {
  model: string;
  prevSummary: string;
  userText: string;
  assistantText: string;
  maxTokens: number;
}): Promise<string | null> {
  const prev = safeOneLine(args.prevSummary || "");
  const user = safeOneLine(args.userText);
  const asst = safeOneLine(args.assistantText);
  const content = [
    `Previous summary: ${prev || "(empty)"}`,
    "Latest exchange:",
    `User: ${user}`,
    `Assistant: ${asst}`,
    "Write the updated one-line summary:",
  ].join("\n");

  const resp = await openai.responses.create({
    model: args.model,
    input: [
      { role: "system", content: summaryUpdateSystemPrompt() },
      { role: "user", content },
    ],
    temperature: 0,
    max_output_tokens: clamp(20, 120, Math.round(args.maxTokens)),
  });

  const { text } = extractFirstTextAndLogprobs(resp);
  const one = safeOneLine(text);
  if (!one) return null;
  // hard cap (safety belt)
  return one.length > 160 ? one.slice(0, 160) : one;
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { sessionId: string; userText: string };
    const sessionId = body?.sessionId;
    const userText = (body?.userText ?? "").toString();
    if (!sessionId || !userText.trim()) {
      return new Response(JSON.stringify({ error: "Missing sessionId or userText" }), { status: 400 });
    }

    const model = process.env.OPENAI_MODEL || "gpt-4.1";

    const sess = getSession(sessionId);
    sess.turn += 1;
    const turn = sess.turn;

    // append user message to history
    sess.history.push({ role: "user", content: userText });

    // ----------------
    // Phase A: PROBE (same model)
    // ----------------
    const probeResp = await openai.responses.create({
      model,
      input: [
        { role: "system", content: probeSystemPrompt() },
        { role: "user", content: userText },
      ],
      temperature: 0.2,
      max_output_tokens: 90,
      // request logprobs
      include: ["message.output_text.logprobs"],
      top_logprobs: 20,
    });

    const { text: probeText, logprobs: probeLogprobs } = extractFirstTextAndLogprobs(probeResp);
    const probeFields = parseProbeFields(probeText);
    const originalProbe: ProbeFields = { raw: probeText || "", ...probeFields };

    // Metrics from first line only (anti self-intoxication)
    const firstLine = sliceFirstLine(probeLogprobs, 30);
    const S = meanSurprisal(firstLine);
    const H = approxEntropy(firstLine);

    let zS: number | null = null;
    let zH: number | null = null;
    let score: number | null = null;
    const notes: string[] = [];

    // fallback: keep previous
    let rawStateFromScore = sess.gate.last_state;

    const ETA = 0.06; // baseline update speed

    if (S !== null) {
      zS = (S - sess.gate.S.mean) / Math.sqrt(sess.gate.S.var + 1e-8);
      sess.gate.S.mean = (1 - ETA) * sess.gate.S.mean + ETA * S;
      const d = S - sess.gate.S.mean;
      sess.gate.S.var = (1 - ETA) * sess.gate.S.var + ETA * d * d;
    } else {
      notes.push("logprobs missing → surprisal unavailable");
    }

    if (H !== null) {
      zH = (H - sess.gate.H.mean) / Math.sqrt(sess.gate.H.var + 1e-8);
      sess.gate.H.mean = (1 - ETA) * sess.gate.H.mean + ETA * H;
      const d = H - sess.gate.H.mean;
      sess.gate.H.var = (1 - ETA) * sess.gate.H.var + ETA * d * d;
    } else {
      notes.push("top_logprobs missing → entropy unavailable");
    }

    if (zS !== null || zH !== null) {
      const a = 0.7;
      const b = 0.3;
      const sPart = zS ?? 0;
      const hPart = zH ?? 0;
      score = a * sPart + b * hPart;
      const tau = 1.2;
      rawStateFromScore = 1 / (1 + Math.exp(-score / tau));
    }

    // ----------------
    // Phase A.1: Exploration Pulse (optional)
    // ----------------
    const stagnationDetected = shouldExplorationPulse(sess.gate);
    const repeatingDim = sess.gate.last_dims.length ? sess.gate.last_dims[sess.gate.last_dims.length - 1] : null;
    const cooldownOk = turn - sess.gate.last_pulse_turn >= 6;
    const notRiskLoop = repeatingDim !== "RISK";
    const boredomish = sess.gate.last_state < 0.6; // don't explore when we're already in high-compute mode

    let effectiveProbe: ProbeFields = originalProbe;
    const pulse: PulseInfo = {
      triggered: false,
      stagnation_detected: stagnationDetected,
      repeating_dim: repeatingDim,
      candidates_text: null,
      picked: null,
      selected_probe: null,
    };

    if (stagnationDetected && cooldownOk && notRiskLoop && boredomish) {
      try {
        // Generate 3 alternative probes (high temp, short)
        const exploreResp = await openai.responses.create({
          model,
          input: [
            { role: "system", content: explorationSystemPrompt(repeatingDim) },
            {
              role: "user",
              content: [
                "User message:",
                userText,
                "\nOriginal probe (current frame):",
                originalProbe.raw,
              ].join("\n"),
            },
          ],
          temperature: 0.95,
          max_output_tokens: 220,
        });

        const { text: candidatesText } = extractFirstTextAndLogprobs(exploreResp);
        pulse.candidates_text = candidatesText || null;

        const cands = parseCandidates(candidatesText || "");
        if (cands.length >= 2) {
          // Verify pick (temp ~0)
          const verifyResp = await openai.responses.create({
            model,
            input: [
              { role: "system", content: verifyPickSystemPrompt() },
              {
                role: "user",
                content: [
                  "User message:",
                  userText,
                  "\nOriginal probe:",
                  originalProbe.raw,
                  "\nCandidates:",
                  cands
                    .map((c, i) => `Candidate ${i + 1}:\n${c.raw}`)
                    .join("\n\n"),
                ].join("\n"),
              },
            ],
            temperature: 0,
            max_output_tokens: 20,
          });

          const { text: pickText } = extractFirstTextAndLogprobs(verifyResp);
          const pick = parsePick(pickText || "") ?? 1;
          const idx = clamp(1, cands.length, pick) - 1;
          const selected = cands[idx];

          pulse.triggered = true;
          pulse.picked = idx + 1;
          pulse.selected_probe = selected.raw;
          sess.gate.last_pulse_turn = turn;

          effectiveProbe = selected;
          notes.push("exploration pulse → selected alternate DIM");
        }
      } catch (e: any) {
        notes.push(`exploration pulse failed: ${String(e?.message ?? e)}`);
      }
    }

    // ----------------
    // state: score → (DIM weight) → hysteresis
    // ----------------
    const dimWeighted = applyDimWeight(rawStateFromScore, effectiveProbe.dim, sess.gate);
    notes.push(...dimWeighted.notes);

    const state = hysteresisUpdate(sess.gate.last_state, dimWeighted.raw, 0.62, 0.48, 0.6);
    sess.gate.last_state = state;

    // Update stagnation buffers using the EFFECTIVE probe (so pulses actually break repetition)
    updateStagnationBuffers(sess.gate, effectiveProbe.dim, effectiveProbe.focus, state, 12);

    // META cap: staged relaxation (0.55 → 0.65 → unlocked) when META isn't dominating.
    const metaRelax = maybeRelaxMetaCapStage(sess.gate, turn);
    if (metaRelax.changed) {
      const capNow = metaCapValue(sess.gate.meta_cap_stage);
      const capLabel = capNow === null ? "unlocked" : String(capNow);
      notes.push(
        `META cap relaxed → stage ${sess.gate.meta_cap_stage} (cap ${capLabel})` +
          (metaRelax.metaShare !== null ? `, recent META share=${metaRelax.metaShare.toFixed(2)}` : "")
      );
    }

    // Update attention log
    if (effectiveProbe.dim && effectiveProbe.focus) {
      const entry: AttentionLogEntry = {
        turn,
        dim: effectiveProbe.dim,
        focus: effectiveProbe.focus,
        next: effectiveProbe.next,
      };
      sess.memory.attn_log.push(entry);
      while (sess.memory.attn_log.length > 30) sess.memory.attn_log.shift();
    }

    // ----------------
    // Salience-based fragment memory bank ("local notes")
    // - decay depends on state (low state -> faster forgetting)
    // - add/merge a short fragment each turn from the attention frame
    // - later we'll inject top-K fragments (not just the latest ones)
    // ----------------
    const decay = decayFragments(sess.memory.fragments, state);
    let fragAdd: {
      added: boolean;
      merged: boolean;
      id: string;
      salience: number;
      text: string;
    } | null = null;

    const baseFragText = fragmentText({
      dim: effectiveProbe.dim,
      focus: effectiveProbe.focus,
      next: effectiveProbe.next,
    });

    if (baseFragText) {
      const initSalience = clamp(0, 1.2, 0.25 + 0.55 * state + dimBonus(effectiveProbe.dim));
      const key = makeFragmentKey({ dim: effectiveProbe.dim, focus: effectiveProbe.focus, next: effectiveProbe.next });
      const res = upsertFragment(
        sess.memory.fragments,
        {
          key,
          turn,
          dim: effectiveProbe.dim,
          focus: effectiveProbe.focus,
          text: baseFragText,
          salience: initSalience,
        },
        turn
      );
      fragAdd = { ...res, salience: initSalience, text: baseFragText };
    }

    pruneFragments(sess.memory.fragments, turn, 40);

    // ----------------
    // Gradient memory budgets (continuous; no discrete LOW/MID/HIGH)
    // ----------------
    const ctx_keep_msgs = clamp(4, 18, Math.round(lerp(4, 18, powEase(state, 1.25))));
    const summary_chars = Math.round(lerp(0, 220, smoothstep(0.22, 0.62, state)));
    const attn_items = clamp(0, 10, Math.round(lerp(0, 10, smoothstep(0.35, 0.85, state))));
    // new: top-K salience fragments (not necessarily recent)
    let frag_items = clamp(0, 10, Math.round(lerp(0, 10, smoothstep(0.20, 0.86, state))));

    // even in low state, keep at least 1 fragment if something is very salient
    const topSalience = sess.memory.fragments.length
      ? Math.max(...sess.memory.fragments.map((f) => f.salience))
      : 0;
    if (frag_items === 0 && topSalience > 0.9) frag_items = 1;

    // Summary is extra compute; schedule it smoothly with state too.
    const summary_update_interval = clamp(1, 18, Math.round(lerp(18, 1, powEase(state, 1.4))));
    const summary_update_max_tokens = clamp(20, 120, Math.round(lerp(25, 90, powEase(state, 1.2))));

    // ----------------
    // Map state → generation params
    // ----------------
    let max_output_tokens = Math.round(lerp(60, 520, state));
    const temperature = clamp(0, 1.2, lerp(0.05, 0.7, state));

    // If exploration pulse just changed the frame, ensure the main response has enough room
    // to make the "視点スライド" feel tangible.
    if (pulse.triggered) {
      const floor = 120;
      if (max_output_tokens < floor) notes.push(`pulse → max_output_tokens floor ${floor}`);
      max_output_tokens = Math.max(max_output_tokens, floor);
    }

    // ----------------
    // Phase B: MAIN
    // ----------------
    const ctx = sess.history.slice(-ctx_keep_msgs);

    const sysParts: { role: "system"; content: string }[] = [{ role: "system", content: mainSystemPrompt(state) }];

    // Gradient memory injection (scaled by state)
    let summaryUsed: string | null = null;
    if (sess.memory.summary && summary_chars >= 40) {
      summaryUsed =
        sess.memory.summary.length > summary_chars ? sess.memory.summary.slice(0, summary_chars) : sess.memory.summary;
      sysParts.push({ role: "system", content: `SUMMARY: ${summaryUsed}` });
    }

    if (attn_items > 0 && sess.memory.attn_log.length) {
      sysParts.push({ role: "system", content: formatAttnLog(sess.memory.attn_log, attn_items) });
    }

    // Inject salience-ranked memory fragments (can include older-but-important notes)
    const fragPicked = pickTopFragments(sess.memory.fragments, frag_items);
    if (fragPicked.length) {
      sysParts.push({ role: "system", content: formatFragmentsForPrompt(fragPicked) });
      rehearseFragments(sess.memory.fragments, fragPicked, turn);
    }

    // Always add current attention frame
    sysParts.push({
      role: "system",
      content: frameSystemPrompt({
        dim: effectiveProbe.dim,
        focus: effectiveProbe.focus,
        next: effectiveProbe.next,
        pulse: pulse.triggered,
      }),
    });

    const mainResp = await openai.responses.create({
      model,
      input: [...sysParts, ...ctx.map((m) => ({ role: m.role, content: m.content }))],
      temperature,
      max_output_tokens,
    });

    const { text: assistantText } = extractFirstTextAndLogprobs(mainResp);

    sess.history.push({ role: "assistant", content: assistantText });

    // ----------------
    // Phase B.1: Update running summary (scheduled smoothly by state)
    // ----------------
    const shouldUpdateSummary = turn - sess.memory.summary_updated_turn >= summary_update_interval;
    if (shouldUpdateSummary) {
      try {
        const newSummary = await updateOneLineSummary({
          model,
          prevSummary: sess.memory.summary,
          userText,
          assistantText,
          maxTokens: summary_update_max_tokens,
        });
        if (newSummary) {
          sess.memory.summary = newSummary;
          sess.memory.summary_updated_turn = turn;
        }
      } catch (e: any) {
        notes.push(`summary update failed: ${String(e?.message ?? e)}`);
      }
    }

    const debug = {
      probeText: effectiveProbe.raw || null,
      probeText_original: originalProbe.raw || null,
      dim: effectiveProbe.dim,
      focus: effectiveProbe.focus,
      next: effectiveProbe.next,
      memory: {
        ctx_keep_msgs,
        summary_chars,
        attn_items,
        frag_items,
        summary_update_interval,
        summary_update_max_tokens,
        fragments: {
          total: sess.memory.fragments.length,
          decay_factor: decay.factor,
          top_salience: topSalience,
          last_add: fragAdd,
          injected: fragPicked.map((f) => ({
            id: f.id,
            turn: f.turn,
            dim: f.dim,
            salience: f.salience,
            text: f.text,
          })),
          top: pickTopFragments(sess.memory.fragments, Math.min(10, sess.memory.fragments.length)).map((f) => ({
            id: f.id,
            turn: f.turn,
            dim: f.dim,
            salience: f.salience,
            text: f.text,
          })),
        },
      },
      pulse,
      summary_used: summaryUsed,
      summary_stored: sess.memory.summary || null,
      metrics: {
        surprisal: S,
        entropy: H,
        score,
      },
      state,
      meta: {
        meta_cap_stage: sess.gate.meta_cap_stage,
        meta_cap: metaCapValue(sess.gate.meta_cap_stage),
        recent_meta_share: recentMetaShare(sess.gate, 12),
      },
      params: {
        max_output_tokens,
        temperature,
        context_keep_msgs: ctx_keep_msgs,
      },
      notes,
    };

    return new Response(JSON.stringify({ assistantText, debug }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: String(err?.message ?? err) }), { status: 500 });
  }
}
