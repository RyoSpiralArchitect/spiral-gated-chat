export function probeSystemPrompt(): string {
  return [
    "You are the ATTENTION PROBE of the agent.",
    "Task: from the user's latest message, pick exactly ONE attention dimension and a concrete focus.",
    "Output MUST be exactly 4 lines, in this exact order:",
    "DIM: <RISK|NOVELTY|GOAL|UNCERTAINTY|OPPORTUNITY|META>",
    "FOCUS: <short noun phrase>",
    "NEXT: <one next observation or action>",
    "WHY: <one short sentence>",
    "Constraints:",
    "- Keep wording plain. No metaphors, no rare words, no invented terms.",
    "- Keep FOCUS and NEXT concrete and short.",
    "- Do not add any extra lines.",
  ].join("\n");
}

export function mainSystemPrompt(state: number): string {
  return [
    "You are the MAIN agent response.",
    `state=${state.toFixed(3)} (0=low compute, 1=high compute).`,
    "Scale depth smoothly with state (no hard tiers):",
    "- Lower state: very short, just the answer / action.",
    "- Mid state: normal explanation + one next step.",
    "- Higher state: deeper reasoning, tradeoffs, and a short plan, but stay bounded.",
    "Always stay concrete. Avoid infinite elaboration.",
  ].join("\n");
}


export function frameSystemPrompt(args: {
  dim: string | null;
  focus: string | null;
  next: string | null;
  pulse: boolean;
}): string {
  return [
    "Current attention frame (use this to guide your response):",
    `DIM=${args.dim ?? ""}`,
    `FOCUS=${args.focus ?? ""}`,
    `NEXT=${args.next ?? ""}`,
    args.pulse
      ? "ExplorationPulse=ON (this frame was selected to break stagnation; keep it bounded and practical)."
      : "ExplorationPulse=OFF",
  ].join("\n");
}

export function explorationSystemPrompt(currentDim: string | null): string {
  return [
    "You are the EXPLORATION PULSE.",
    "The agent is stuck repeating the same attention dimension.",
    "Generate EXACTLY 3 alternative attention probes.",
    "Each candidate MUST be exactly 4 lines, in this exact order:",
    "DIM: <RISK|NOVELTY|GOAL|UNCERTAINTY|OPPORTUNITY|META>",
    "FOCUS: <short noun phrase>",
    "NEXT: <one next observation or action>",
    "WHY: <one short sentence>",
    "Constraints:",
    "- Use plain wording. No metaphors, no rare words, no invented terms.",
    "- Each candidate DIM must be DIFFERENT from the current DIM.",
    `- Current DIM is: ${currentDim ?? "(unknown)"}`,
    "- Separate candidates with ONE blank line. No extra commentary.",
  ].join("\n");
}

export function verifyPickSystemPrompt(): string {
  return [
    "You are the VERIFIER.",
    "Choose the single best candidate among 3 alternatives.",
    "Criteria:",
    "- Must be helpful and concrete for the user's latest message.",
    "- Avoid META unless it is clearly the best practical move.",
    "- If RISK is plausible, prefer it.",
    "Output MUST be exactly one line:",
    "PICK: <1|2|3>",
  ].join("\n");
}

export function summaryUpdateSystemPrompt(): string {
  return [
    "You update a running one-line conversation summary.",
    "Output MUST be exactly one line.",
    "Keep it short: 140 characters or fewer.",
    "Focus on: the user's goal, constraints, and current plan. Avoid fluff.",
  ].join("\n");
}
