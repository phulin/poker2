import { For, Show, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Play,
  Plus,
  RotateCcw,
  Trash2,
  X,
} from "lucide-solid";
import {
  BetterFfnWebGpuModel,
  createBrowserCfrEvaluator,
  createBrowserDevice,
} from "./browser.js";
import { parseBetterFfnManifest, resolveCfrDefaults } from "./modelFormat.js";
import { loadModelBytesWithCache, type ModelCacheProgress } from "./modelCache.js";
import { formatCard, handComboIndex, handComboCards } from "./cards.js";
import { PublicHunlEnv, NUM_HANDS, type LegalBins } from "./hunlEnv.js";
import type {
  BetterFfnManifest,
  BrowserEvaluationResult,
  PlayerIndex,
} from "./types.js";

const MODEL_MANIFEST_URL = "/models/rebel_latest/model.json";
const CARD_OPTIONS = Array.from({ length: 52 }, (_, index) => formatCard(index));
const STREET_CARD_COUNTS = [0, 3, 4, 5] as const;

interface Runtime {
  device: GPUDevice;
  model: BetterFfnWebGpuModel;
  evaluator: ReturnType<typeof createBrowserCfrEvaluator>;
  manifest: BetterFfnManifest;
  cached: boolean;
}

interface ActionContext {
  amounts: number[];
  toCall: number;
  meCommitted: number;
  stack: number;
  allInIndex: number;
}

interface ActionRow {
  action: number;
  actor: PlayerIndex;
  legalMask: number[];
  context: ActionContext;
}

interface StateDescriptor {
  rows: ActionRow[];
  finalActor: PlayerIndex;
  finalLegalMask: number[];
  finalContext: ActionContext;
}

interface SolveResult {
  result: BrowserEvaluationResult;
  elapsedMs: number;
  iterations: number;
  depth: number;
  cfrAvg: boolean;
  heroHandIndex: number;
  villainSummary: RangeSummary;
}

interface RangeSummary {
  mass: number;
  combos: number;
  top: Array<{ hand: string; weight: number }>;
}

function asPlayer(value: string): PlayerIndex {
  return value === "1" ? 1 : 0;
}

function contextFromEnv(env: PublicHunlEnv, legal: LegalBins): ActionContext {
  const me = env.toAct;
  const opp = (1 - me) as PlayerIndex;
  return {
    amounts: [...legal.amounts],
    toCall: env.committed[opp] - env.committed[me],
    meCommitted: env.committed[me],
    stack: env.stacks[me],
    allInIndex: legal.amounts.length - 1,
  };
}

function formatChips(amount: number): string {
  if (!Number.isFinite(amount)) return "?";
  const rounded = Math.round(amount * 100) / 100;
  if (Number.isInteger(rounded)) return rounded.toString();
  return rounded.toFixed(2).replace(/\.?0+$/, "");
}

function formatActionLabel(bin: number, ctx: ActionContext): string {
  if (bin === 0) return "Fold";
  if (bin === 1) {
    return ctx.toCall > 0 ? `Call ${formatChips(ctx.toCall)}` : "Check";
  }
  if (bin === ctx.allInIndex) return "All-in";
  const amount = ctx.amounts[bin] ?? -1;
  if (amount < 0) return `Bin ${bin}`;
  if (amount >= ctx.stack) return "All-in";
  if (ctx.toCall > 0) {
    const raiseTo = ctx.meCommitted + amount;
    return `Raise to ${formatChips(raiseTo)}`;
  }
  return `Bet ${formatChips(amount)}`;
}

function positiveNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = ["s", "h", "d", "c"] as const;
const SUIT_GLYPHS: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };

function suitClass(suit: string): string {
  if (suit === "h") return "suit-h";
  if (suit === "d") return "suit-d";
  if (suit === "c") return "suit-c";
  return "suit-s";
}

function CardChip(props: { value: string; placeholder?: string }): JSX.Element {
  return (
    <Show
      when={props.value}
      fallback={<span class="card-chip empty">{props.placeholder ?? "+"}</span>}
    >
      <span class={`card-chip ${suitClass(props.value[1] ?? "s")}`}>
        <span class="card-rank">{props.value[0]}</span>
        <span class="card-suit">{SUIT_GLYPHS[props.value[1] ?? "s"]}</span>
      </span>
    </Show>
  );
}

function CardPicker(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: ReadonlySet<string>;
  allowEmpty?: boolean;
}): JSX.Element {
  const [open, setOpen] = createSignal(false);

  function close() {
    setOpen(false);
  }

  function select(card: string) {
    props.onChange(card);
    close();
  }

  return (
    <div class="card-picker">
      <span class="card-picker-label">{props.label}</span>
      <button
        type="button"
        class="card-picker-trigger"
        onClick={() => setOpen((current) => !current)}
      >
        <CardChip value={props.value} />
      </button>
      <Show when={open()}>
        <div class="card-picker-backdrop" onClick={close} />
        <div class="card-picker-popover" role="dialog">
          <div class="card-picker-header">
            <span>Pick a card</span>
            <button type="button" class="icon-button small" onClick={close} title="Close">
              <X size={14} />
            </button>
          </div>
          <div class="card-picker-grid">
            <For each={SUITS}>
              {(suit) => (
                <For each={RANKS}>
                  {(rank) => {
                    const card = `${rank}${suit}`;
                    const taken = () => props.disabled?.has(card) && card !== props.value;
                    const selected = () => props.value === card;
                    return (
                      <button
                        type="button"
                        class={
                          "card-cell " +
                          suitClass(suit) +
                          (selected() ? " selected" : "") +
                          (taken() ? " taken" : "")
                        }
                        disabled={taken()}
                        onClick={() => select(card)}
                      >
                        <span class="card-rank">{rank}</span>
                        <span class="card-suit">{SUIT_GLYPHS[suit]}</span>
                      </button>
                    );
                  }}
                </For>
              )}
            </For>
          </div>
          <Show when={props.allowEmpty || props.value}>
            <button
              type="button"
              class="card-picker-clear"
              onClick={() => {
                props.onChange("");
                close();
              }}
            >
              Clear
            </button>
          </Show>
        </div>
      </Show>
    </div>
  );
}

function App(): JSX.Element {
  const [runtime, setRuntime] = createSignal<Runtime>();
  const [modelProgress, setModelProgress] = createSignal<ModelCacheProgress>({
    phase: "manifest",
    message: "Checking WebGPU",
  });
  const [modelError, setModelError] = createSignal("");
  const [heroPlayer, setHeroPlayer] = createSignal<PlayerIndex>(1);
  const [button, setButton] = createSignal<PlayerIndex>(1);
  const [stack, setStack] = createSignal("200");
  const [sb, setSb] = createSignal("1");
  const [bb, setBb] = createSignal("2");
  const [street, setStreet] = createSignal(0);
  const [iterations, setIterations] = createSignal("16");
  const [depth, setDepth] = createSignal("1");
  const [cfrAvg, setCfrAvg] = createSignal(true);
  const [heroCards, setHeroCards] = createSignal<[string, string]>(["As", "Kd"]);
  const [boardCards, setBoardCards] = createSignal(["", "", "", "", ""]);
  const [actions, setActions] = createSignal<number[]>([]);
  const [solveStatus, setSolveStatus] = createSignal("");
  const [solveError, setSolveError] = createSignal("");
  const [solveResult, setSolveResult] = createSignal<SolveResult>();

  onMount(() => {
    void loadRuntime();
  });

  onCleanup(() => {
    const current = runtime();
    current?.evaluator.dispose();
    current?.model.dispose();
    current?.device.destroy();
  });

  async function loadRuntime(): Promise<void> {
    try {
      setModelProgress({ phase: "manifest", message: "Requesting WebGPU device" });
      const device = await createBrowserDevice();
      const loaded = await loadModelBytesWithCache(MODEL_MANIFEST_URL, {
        onProgress: setModelProgress,
      });
      setModelProgress({ phase: "manifest", message: "Creating WebGPU model" });
      const manifest = parseBetterFfnManifest(loaded.manifest);
      const cfrDefaults = resolveCfrDefaults(manifest);
      setIterations(String(cfrDefaults.iterations));
      setDepth(String(cfrDefaults.depth));
      setCfrAvg(cfrDefaults.cfrAvg);
      const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, loaded.weights);
      const evaluator = createBrowserCfrEvaluator(device, model);
      setRuntime({ device, model, evaluator, manifest, cached: loaded.cached });
      setModelProgress({
        phase: loaded.cached ? "cache-hit" : "stored",
        message: loaded.cached ? "Model loaded from IndexedDB" : "Model loaded and cached",
      });
    } catch (error) {
      setModelError(error instanceof Error ? error.message : String(error));
    }
  }

  const parsedInputs = createMemo(() => {
    try {
      const hero = heroCards();
      if (!hero[0] || !hero[1]) throw new Error("Select both hero cards");
      const publicCount = STREET_CARD_COUNTS[street()] ?? 0;
      const visibleBoard = boardCards().slice(0, publicCount);
      if (visibleBoard.some((card) => card === "")) {
        throw new Error("Select every board card for the chosen street");
      }
      const heroHand = [
        CARD_OPTIONS.includes(hero[0]) ? cardFromOption(hero[0]) : -1,
        CARD_OPTIONS.includes(hero[1]) ? cardFromOption(hero[1]) : -1,
      ] as [number, number];
      const publicCards = visibleBoard.map((card) => cardFromOption(card));
      const seen = new Set<number>();
      for (const card of [...heroHand, ...publicCards]) {
        if (card < 0) throw new Error("Invalid card selection");
        if (seen.has(card)) throw new Error(`Duplicate card ${formatCard(card)}`);
        seen.add(card);
      }
      return { heroHand, publicCards };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  });

  const descriptor = createMemo<StateDescriptor | { error: string }>(() => {
    const current = runtime();
    const cards = parsedInputs();
    if (!current) return { error: "Model is not ready" };
    if ("error" in cards) return { error: cards.error };
    try {
      const env = PublicHunlEnv.fromManifest(current.manifest, {
        button: button(),
        stack: positiveNumber(stack(), current.manifest.env.stack),
        sb: positiveNumber(sb(), current.manifest.env.sb),
        bb: positiveNumber(bb(), current.manifest.env.bb),
      });
      env.configureKnownCards({
        publicCards: cards.publicCards,
        heroPlayer: heroPlayer(),
        heroHand: cards.heroHand,
      });

      const rows: ActionRow[] = [];
      for (const action of actions()) {
        const legal = env.legalBinsAmountAndMask();
        rows.push({
          action,
          actor: env.toAct,
          legalMask: legal.mask,
          context: contextFromEnv(env, legal),
        });
        if (!legal.mask[action]) {
          throw new Error(`Action ${action} is no longer legal`);
        }
        env.stepBin(action, legal);
      }
      const finalLegal = env.legalBinsAmountAndMask();
      return {
        rows,
        finalActor: env.toAct,
        finalLegalMask: finalLegal.mask,
        finalContext: contextFromEnv(env, finalLegal),
      };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  });

  const usedCards = createMemo<Set<string>>(() => {
    const used = new Set<string>();
    for (const card of heroCards()) {
      if (card) used.add(card);
    }
    const publicCount = STREET_CARD_COUNTS[street()] ?? 0;
    for (let i = 0; i < publicCount; i += 1) {
      const card = boardCards()[i];
      if (card) used.add(card);
    }
    return used;
  });

  const manifestActionLabels = createMemo(() => runtime()?.manifest.actionLabels ?? []);
  const descriptorError = createMemo(() => {
    const current = descriptor();
    return "error" in current ? current.error : "";
  });
  const modelProgressPercent = createMemo(() => {
    const progress = modelProgress();
    if (!progress.totalBytes || progress.loadedBytes === undefined) return undefined;
    return Math.min(100, (progress.loadedBytes / progress.totalBytes) * 100);
  });

  function cardFromOption(value: string): number {
    const index = CARD_OPTIONS.indexOf(value);
    if (index < 0) throw new Error(`Invalid card ${value}`);
    return index;
  }

  function updateHeroCard(index: 0 | 1, value: string): void {
    setHeroCards((cards) => {
      const next: [string, string] = [...cards];
      next[index] = value;
      return next;
    });
    setSolveResult(undefined);
  }

  function updateBoardCard(index: number, value: string): void {
    setBoardCards((cards) => {
      const next = [...cards];
      next[index] = value;
      return next;
    });
    setSolveResult(undefined);
  }

  function setActionAt(index: number, action: number): void {
    setActions((current) => current.map((value, i) => (i === index ? action : value)));
    setSolveResult(undefined);
  }

  function removeAction(index: number): void {
    setActions((current) => current.filter((_, i) => i !== index));
    setSolveResult(undefined);
  }

  function addAction(action: number): void {
    setActions((current) => [...current, action]);
    setSolveResult(undefined);
  }

  async function solve(): Promise<void> {
    const current = runtime();
    const cards = parsedInputs();
    const state = descriptor();
    if (!current || "error" in cards || "error" in state) return;

    try {
      setSolveError("");
      const solveIterations = Math.max(1, Math.trunc(positiveNumber(iterations(), 16)));
      const solveDepth = Math.max(1, Math.trunc(positiveNumber(depth(), 1)));
      const solveCfrAvg = cfrAvg();
      setSolveStatus(`Solving depth ${solveDepth} for ${solveIterations} CFR iterations`);
      setSolveResult(undefined);
      const started = performance.now();
      const result = await current.evaluator.evaluateSpot({
        spot: actions(),
        iterations: solveIterations,
        depth: solveDepth,
        cfrAvg: solveCfrAvg,
        initialState: {
          button: button(),
          stack: positiveNumber(stack(), current.manifest.env.stack),
          sb: positiveNumber(sb(), current.manifest.env.sb),
          bb: positiveNumber(bb(), current.manifest.env.bb),
        },
        publicCards: cards.publicCards,
        heroPlayer: heroPlayer(),
        heroHand: cards.heroHand,
      });
      const elapsedMs = performance.now() - started;
      const heroHandIndex = handComboIndex(cards.heroHand[0], cards.heroHand[1]);
      setSolveResult({
        result,
        elapsedMs,
        iterations: solveIterations,
        depth: solveDepth,
        cfrAvg: solveCfrAvg,
        heroHandIndex,
        villainSummary: summarizeVillainRange(result.beliefsAtSpot, heroPlayer()),
      });
      setSolveStatus("Solve complete");
    } catch (error) {
      setSolveError(error instanceof Error ? error.message : String(error));
      setSolveStatus("");
    }
  }

  function actionName(index: number, context?: ActionContext): string {
    if (context) {
      return formatActionLabel(index, context);
    }
    return manifestActionLabels()[index] ?? `action_${index}`;
  }

  function legalActions(mask: readonly number[]): number[] {
    const actionsOut: number[] = [];
    for (let i = 0; i < mask.length; i += 1) {
      if (mask[i]) actionsOut.push(i);
    }
    return actionsOut;
  }

  return (
    <main class="app-shell">
      <header class="topbar">
        <div>
          <h1>WebGPU CFR Spot Solver</h1>
          <p>BetterFFN local re-solve for heads-up no-limit Hold 'Em</p>
        </div>
        <div class="model-status">
          <Cpu size={18} />
          <span>{modelError() || modelProgress().message}</span>
        </div>
      </header>

      <Show when={modelProgressPercent() !== undefined}>
        <div class="progress-track" aria-label="Model download progress">
          <div style={{ width: `${modelProgressPercent() ?? 0}%` }} />
        </div>
      </Show>

      <Show when={modelError()}>
        <div class="notice error">
          <AlertTriangle size={18} />
          <span>{modelError()}</span>
        </div>
      </Show>

      <section class="workspace">
        <form class="panel controls" onSubmit={(event) => event.preventDefault()}>
          <div class="section-head">
            <h2>Configure Spot</h2>
            <button
              type="button"
              class="icon-button"
              title="Reset actions"
              onClick={() => {
                setActions([]);
                setSolveResult(undefined);
              }}
            >
              <RotateCcw size={16} />
            </button>
          </div>

          <div class="subgroup">
            <span class="subgroup-title">Game</span>
            <div class="grid two">
              <label class="field">
                <span>Hero seat</span>
                <select
                  value={String(heroPlayer())}
                  onChange={(event) => {
                    setHeroPlayer(asPlayer(event.currentTarget.value));
                    setSolveResult(undefined);
                  }}
                >
                  <option value="0">Player 0</option>
                  <option value="1">Player 1</option>
                </select>
              </label>
              <label class="field">
                <span>Button (small blind)</span>
                <select
                  value={String(button())}
                  onChange={(event) => {
                    setButton(asPlayer(event.currentTarget.value));
                    setActions([]);
                    setSolveResult(undefined);
                  }}
                >
                  <option value="0">Player 0</option>
                  <option value="1">Player 1</option>
                </select>
              </label>
            </div>
            <div class="grid three">
              <label class="field">
                <span>Stack</span>
                <input value={stack()} inputmode="decimal" onInput={(event) => setStack(event.currentTarget.value)} />
              </label>
              <label class="field">
                <span>Small blind</span>
                <input value={sb()} inputmode="decimal" onInput={(event) => setSb(event.currentTarget.value)} />
              </label>
              <label class="field">
                <span>Big blind</span>
                <input value={bb()} inputmode="decimal" onInput={(event) => setBb(event.currentTarget.value)} />
              </label>
            </div>
          </div>

          <div class="card-section">
            <div class="card-section-head">
              <span class="card-section-title">Hero hand</span>
              <div class="street-toggle" role="tablist">
                <For each={["Preflop", "Flop", "Turn", "River"]}>
                  {(name, index) => (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={street() === index()}
                      class={street() === index() ? "active" : ""}
                      onClick={() => {
                        setStreet(index());
                        setActions([]);
                        setSolveResult(undefined);
                      }}
                    >
                      {name}
                    </button>
                  )}
                </For>
              </div>
            </div>
            <div class="card-row">
              <CardPicker
                label="Hero 1"
                value={heroCards()[0]}
                disabled={usedCards()}
                onChange={(value) => updateHeroCard(0, value)}
              />
              <CardPicker
                label="Hero 2"
                value={heroCards()[1]}
                disabled={usedCards()}
                onChange={(value) => updateHeroCard(1, value)}
              />
            </div>
            <Show when={(STREET_CARD_COUNTS[street()] ?? 0) > 0}>
              <span class="card-section-title">Board</span>
              <div class="card-row">
                <For each={Array.from({ length: STREET_CARD_COUNTS[street()] ?? 0 }, (_, index) => index)}>
                  {(index) => (
                    <CardPicker
                      label={`Board ${index + 1}`}
                      value={boardCards()[index] ?? ""}
                      disabled={usedCards()}
                      allowEmpty
                      onChange={(value) => updateBoardCard(index, value)}
                    />
                  )}
                </For>
              </div>
            </Show>
          </div>

          <div class="actions-editor">
            <h3>Action Sequence</h3>
            <Show when={!descriptorError()} fallback={<p class="inline-error">{descriptorError()}</p>}>
              <For each={(descriptor() as StateDescriptor).rows}>
                {(row, index) => (
                  <div class="action-row">
                    <span class="actor">P{row.actor}</span>
                    <select
                      value={String(row.action)}
                      onChange={(event) => setActionAt(index(), Number(event.currentTarget.value))}
                    >
                      <For each={legalActions(row.legalMask)}>
                        {(action) => (
                          <option value={String(action)}>{actionName(action, row.context)}</option>
                        )}
                      </For>
                    </select>
                    <button
                      type="button"
                      class="icon-button"
                      title="Remove action"
                      onClick={() => removeAction(index())}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                )}
              </For>
              <div class="add-actions">
                <For each={legalActions((descriptor() as StateDescriptor).finalLegalMask)}>
                  {(action) => (
                    <button type="button" onClick={() => addAction(action)}>
                      <Plus size={15} />
                      <span>{actionName(action, (descriptor() as StateDescriptor).finalContext)}</span>
                    </button>
                  )}
                </For>
              </div>
            </Show>
          </div>

          <div class="subgroup">
            <span class="subgroup-title">Solve</span>
            <div class="grid three">
              <label class="field">
                <span>Iterations</span>
                <input value={iterations()} inputmode="numeric" min="1" step="1" onInput={(event) => setIterations(event.currentTarget.value)} />
              </label>
              <label class="field">
                <span>Depth</span>
                <input value={depth()} inputmode="numeric" min="1" step="1" onInput={(event) => setDepth(event.currentTarget.value)} />
              </label>
              <label class="field toggle-field">
                <span>CFR avg beliefs</span>
                <button
                  type="button"
                  class={`toggle ${cfrAvg() ? "on" : "off"}`}
                  role="switch"
                  aria-checked={cfrAvg()}
                  onClick={() => {
                    setCfrAvg(!cfrAvg());
                    setSolveResult(undefined);
                  }}
                >
                  <span class="toggle-thumb" />
                </button>
              </label>
            </div>
            <button
              type="button"
              class="solve-button"
              disabled={!runtime() || Boolean(descriptorError())}
              onClick={() => void solve()}
            >
              <Play size={18} />
              <span>Solve Spot</span>
            </button>
          </div>
        </form>

        <section class="panel results">
          <div class="section-head">
            <h2>Results</h2>
            <Show when={runtime()}>
              <span class="hash">sha {runtime()!.manifest.weights.sha256.slice(0, 12)}</span>
            </Show>
          </div>

          <Show when={solveStatus()}>
            <div class="notice ok">
              <CheckCircle2 size={18} />
              <span>{solveStatus()}</span>
            </div>
          </Show>
          <Show when={solveError()}>
            <div class="notice error">
              <AlertTriangle size={18} />
              <span>{solveError()}</span>
            </div>
          </Show>

          <Show
            when={solveResult()}
            fallback={<div class="empty-state">Run a solve to populate strategy and range output.</div>}
          >
            {(solved) => (
              <>
                <div class="metadata">
                  <span>{solved().iterations} iterations</span>
                  <span>depth {solved().depth}</span>
                  <span>{solved().cfrAvg ? "cfr-avg beliefs" : "current beliefs"}</span>
                  <span>{solved().elapsedMs.toFixed(1)} ms</span>
                  <span>{runtime()?.cached ? "cached model" : "fresh model"}</span>
                  <span>actor P{solved().result.actor}</span>
                </div>

                <StrategyTable
                  labels={solved().result.actionLabels}
                  legalMask={solved().result.legalMask}
                  actionProbs={solved().result.actionProbs}
                  context={(descriptor() as StateDescriptor).finalContext}
                />

                <HeroPolicy
                  labels={solved().result.actionLabels}
                  legalMask={solved().result.legalMask}
                  policy={solved().result.policy}
                  handIndex={solved().heroHandIndex}
                  actor={solved().result.actor}
                  heroPlayer={heroPlayer()}
                  context={(descriptor() as StateDescriptor).finalContext}
                />

                <RangeSummaryView summary={solved().villainSummary} />
              </>
            )}
          </Show>
        </section>
      </section>
    </main>
  );
}

function StrategyTable(props: {
  labels: readonly string[];
  legalMask: readonly number[];
  actionProbs: Float32Array<ArrayBufferLike>;
  context?: ActionContext;
}): JSX.Element {
  return (
    <table class="strategy-table">
      <thead>
        <tr>
          <th>Action</th>
          <th>Legal</th>
          <th>Strategy</th>
        </tr>
      </thead>
      <tbody>
        <For each={props.labels}>
          {(label, index) => (
            <tr class={props.legalMask[index()] ? "" : "muted-row"}>
              <td>{props.context ? formatActionLabel(index(), props.context) : label}</td>
              <td>{props.legalMask[index()] ? "yes" : "no"}</td>
              <td>{formatPercent(props.actionProbs[index()] ?? 0)}</td>
            </tr>
          )}
        </For>
      </tbody>
    </table>
  );
}

function HeroPolicy(props: {
  labels: readonly string[];
  legalMask: readonly number[];
  policy: Float32Array<ArrayBufferLike>;
  handIndex: number;
  actor: PlayerIndex;
  heroPlayer: PlayerIndex;
  context?: ActionContext;
}): JSX.Element {
  const row = createMemo(() => {
    const offset = props.handIndex * props.labels.length;
    return props.labels.map((label, index) => ({
      label: props.context ? formatActionLabel(index, props.context) : label,
      legal: props.legalMask[index] ?? 0,
      value: props.policy[offset + index] ?? 0,
    }));
  });
  return (
    <section class="subsection">
      <h3>Hero Hand Policy</h3>
      <p class="subtle">
        {props.actor === props.heroPlayer
          ? `Hand index ${props.handIndex}`
          : `Current actor is P${props.actor}; hero is P${props.heroPlayer}`}
      </p>
      <div class="policy-row">
        <For each={row()}>
          {(item) => (
            <div class={item.legal ? "policy-cell" : "policy-cell disabled"}>
              <span>{item.label}</span>
              <strong>{formatPercent(item.value)}</strong>
            </div>
          )}
        </For>
      </div>
    </section>
  );
}

function RangeSummaryView(props: { summary: RangeSummary }): JSX.Element {
  return (
    <section class="subsection">
      <h3>Villain Range</h3>
      <div class="metadata">
        <span>{props.summary.combos} combos</span>
        <span>mass {props.summary.mass.toFixed(4)}</span>
      </div>
      <div class="range-list">
        <For each={props.summary.top}>
          {(item) => (
            <div>
              <span>{item.hand}</span>
              <strong>{formatPercent(item.weight)}</strong>
            </div>
          )}
        </For>
      </div>
    </section>
  );
}

function summarizeVillainRange(
  beliefs: Float32Array<ArrayBufferLike>,
  heroPlayer: PlayerIndex,
): RangeSummary {
  const villain = (1 - heroPlayer) as PlayerIndex;
  const offset = villain * NUM_HANDS;
  const entries: Array<{ hand: string; weight: number }> = [];
  let mass = 0;
  let combos = 0;
  for (let hand = 0; hand < NUM_HANDS; hand += 1) {
    const weight = beliefs[offset + hand] ?? 0;
    mass += weight;
    if (weight > 1.0e-7) {
      combos += 1;
      const [c0, c1] = handComboCards(hand);
      entries.push({ hand: `${formatCard(c0)} ${formatCard(c1)}`, weight });
    }
  }
  entries.sort((a, b) => b.weight - a.weight);
  return { mass, combos, top: entries.slice(0, 8) };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default App;
