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
} from "lucide-solid";
import {
  BetterFfnWebGpuModel,
  createBrowserCfrEvaluator,
  createBrowserDevice,
} from "./browser.js";
import { parseBetterFfnManifest } from "./modelFormat.js";
import { loadModelBytesWithCache, type ModelCacheProgress } from "./modelCache.js";
import { formatCard, handComboIndex, handComboCards } from "./cards.js";
import { PublicHunlEnv, NUM_HANDS } from "./hunlEnv.js";
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

interface ActionRow {
  action: number;
  actor: PlayerIndex;
  legalMask: number[];
}

interface StateDescriptor {
  rows: ActionRow[];
  finalActor: PlayerIndex;
  finalLegalMask: number[];
}

interface SolveResult {
  result: BrowserEvaluationResult;
  elapsedMs: number;
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

function positiveNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function CardSelect(props: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  allowEmpty?: boolean;
}): JSX.Element {
  return (
    <label class="field compact-field">
      <span>{props.label}</span>
      <select value={props.value} onChange={(event) => props.onChange(event.currentTarget.value)}>
        <Show when={props.allowEmpty}>
          <option value="">--</option>
        </Show>
        <For each={CARD_OPTIONS}>{(card) => <option value={card}>{card}</option>}</For>
      </select>
    </label>
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
      if (!navigator.gpu) {
        throw new Error("WebGPU is unavailable in this browser");
      }
      setModelProgress({ phase: "manifest", message: "Requesting WebGPU device" });
      const device = await createBrowserDevice();
      const loaded = await loadModelBytesWithCache(MODEL_MANIFEST_URL, {
        onProgress: setModelProgress,
      });
      setModelProgress({ phase: "manifest", message: "Creating WebGPU model" });
      const manifest = parseBetterFfnManifest(loaded.manifest);
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
        rows.push({ action, actor: env.toAct, legalMask: legal.mask });
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
      };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
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
      setSolveStatus(`Solving ${iterations()} CFR iterations`);
      setSolveResult(undefined);
      const started = performance.now();
      const result = await current.evaluator.evaluateSpot({
        spot: actions(),
        iterations: Math.max(1, Math.trunc(positiveNumber(iterations(), 16))),
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
        heroHandIndex,
        villainSummary: summarizeVillainRange(result.beliefsAtSpot, heroPlayer()),
      });
      setSolveStatus("Solve complete");
    } catch (error) {
      setSolveError(error instanceof Error ? error.message : String(error));
      setSolveStatus("");
    }
  }

  function actionName(index: number): string {
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
            <h2>Spot</h2>
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

          <div class="grid two">
            <label class="field">
              <span>Hero</span>
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
              <span>Button / small blind</span>
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

          <div class="grid four">
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
            <label class="field">
              <span>Iterations</span>
              <input value={iterations()} inputmode="numeric" onInput={(event) => setIterations(event.currentTarget.value)} />
            </label>
          </div>

          <div class="card-grid">
            <CardSelect label="Hero 1" value={heroCards()[0]} onChange={(value) => updateHeroCard(0, value)} />
            <CardSelect label="Hero 2" value={heroCards()[1]} onChange={(value) => updateHeroCard(1, value)} />
            <label class="field compact-field">
              <span>Street</span>
              <select
                value={String(street())}
                onChange={(event) => {
                  setStreet(Number(event.currentTarget.value));
                  setActions([]);
                  setSolveResult(undefined);
                }}
              >
                <option value="0">Preflop</option>
                <option value="1">Flop</option>
                <option value="2">Turn</option>
                <option value="3">River</option>
              </select>
            </label>
            <For each={Array.from({ length: STREET_CARD_COUNTS[street()] ?? 0 }, (_, index) => index)}>
              {(index) => (
                <CardSelect
                  label={`Board ${index + 1}`}
                  value={boardCards()[index] ?? ""}
                  onChange={(value) => updateBoardCard(index, value)}
                />
              )}
            </For>
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
                        {(action) => <option value={String(action)}>{actionName(action)}</option>}
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
                      <span>{actionName(action)}</span>
                    </button>
                  )}
                </For>
              </div>
            </Show>
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
                  <span>{iterations()} iterations</span>
                  <span>{solved().elapsedMs.toFixed(1)} ms</span>
                  <span>{runtime()?.cached ? "cached model" : "fresh model"}</span>
                  <span>actor P{solved().result.actor}</span>
                </div>

                <StrategyTable
                  labels={solved().result.actionLabels}
                  legalMask={solved().result.legalMask}
                  actionProbs={solved().result.actionProbs}
                />

                <HeroPolicy
                  labels={solved().result.actionLabels}
                  legalMask={solved().result.legalMask}
                  policy={solved().result.policy}
                  handIndex={solved().heroHandIndex}
                  actor={solved().result.actor}
                  heroPlayer={heroPlayer()}
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
              <td>{label}</td>
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
}): JSX.Element {
  const row = createMemo(() => {
    const offset = props.handIndex * props.labels.length;
    return props.labels.map((label, index) => ({
      label,
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
