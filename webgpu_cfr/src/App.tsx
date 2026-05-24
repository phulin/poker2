import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Play,
  RotateCcw,
  Trash2,
  X,
} from "lucide-solid";
import {
  BetterFfnWebGpuModel,
  createBrowserCfrEvaluator,
  createBrowserDevice,
} from "./browser.js";
import { parseBetterFfnManifest } from "./modelFormat.js";
import { loadModelBytesWithCache, type ModelCacheProgress } from "./modelCache.js";
import { parseCard, parseCards, formatCard, handComboIndex, handComboCards } from "./cards.js";
import { buildHeroOnlyBeliefs } from "./beliefs.js";
import { PublicHunlEnv, NUM_HANDS, DEFAULT_FORCE_DECK, type LegalBins } from "./hunlEnv.js";
import type {
  BetterFfnManifest,
  BrowserEvaluationResult,
  PlayerIndex,
} from "./types.js";

const MODEL_MANIFEST_URL = "/models/rebel_latest/model.json";
const CARD_OPTIONS = Array.from({ length: 52 }, (_, index) => formatCard(index));
const STREET_CARD_COUNTS = [0, 3, 4, 5] as const;
const STREET_NAMES = ["Preflop", "Flop", "Turn", "River"] as const;
const HERO_PLAYER: PlayerIndex = 0;

interface PublicEnvDefaults {
  stack: number;
  sb: number;
  bb: number;
  betBins: number[];
  flopShowdown: boolean;
  maxStackBb?: number;
  defaultButton: PlayerIndex;
  defaultForceDeck: number[];
}

const LOCAL_ENV_DEFAULTS: PublicEnvDefaults = {
  stack: 400,
  sb: 1,
  bb: 2,
  betBins: [0.25, 0.5, 0.75, 1.0, 1.5],
  flopShowdown: false,
  maxStackBb: 400,
  defaultButton: HERO_PLAYER,
  defaultForceDeck: DEFAULT_FORCE_DECK,
};

interface Runtime {
  device: GPUDevice;
  model: BetterFfnWebGpuModel;
  evaluator: ReturnType<typeof createBrowserCfrEvaluator>;
  manifest: BetterFfnManifest;
  cached: boolean;
  usingSubgroups: boolean;
}

interface ActionContext {
  amounts: number[];
  toCall: number;
  meCommitted: number;
  stack: number;
  allInIndex: number;
  pot: number;
}

interface ActionRow {
  action: number;
  actor: PlayerIndex;
  legalMask: number[];
  context: ActionContext;
  street: number;
}

interface StateDescriptor {
  rows: ActionRow[];
  finalActor: PlayerIndex;
  finalLegalMask: number[];
  finalContext: ActionContext;
  finalStreet: number;
}

interface HashInputs {
  hand?: string | undefined;
  actions?: string | undefined;
  boardCards: string[];
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

function playerLabel(player: PlayerIndex): string {
  return player === HERO_PLAYER ? "Hero" : "Villain";
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
    pot: env.pot,
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

function shortActionLabel(bin: number, ctx: ActionContext): string {
  if (bin === 0) return "F";
  if (bin === 1) return "C";
  if (bin === ctx.allInIndex) return "A";
  return ctx.toCall > 0 ? "R" : "B";
}

function positiveNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function cardFromOption(value: string): number {
  const index = CARD_OPTIONS.indexOf(value);
  if (index < 0) throw new Error(`Invalid card ${value}`);
  return index;
}

function createPublicEnv(
  envDefaults: PublicEnvDefaults,
  options: {
    button: PlayerIndex;
    stack: string;
    sb: string;
    bb: string;
    forceDeck?: number[];
  },
): PublicHunlEnv {
  return new PublicHunlEnv({
    button: options.button,
    stack: positiveNumber(options.stack, envDefaults.stack),
    sb: positiveNumber(options.sb, envDefaults.sb),
    bb: positiveNumber(options.bb, envDefaults.bb),
    betBins: envDefaults.betBins,
    forceDeck: options.forceDeck ?? envDefaults.defaultForceDeck ?? DEFAULT_FORCE_DECK,
    flopShowdown: envDefaults.flopShowdown,
    ...(envDefaults.maxStackBb !== undefined ? { maxStackBb: envDefaults.maxStackBb } : {}),
  });
}

function legalActionList(mask: readonly number[]): number[] {
  const actionsOut: number[] = [];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i]) actionsOut.push(i);
  }
  return actionsOut;
}

function parseBoardHashParam(value: string | null, expectedCount: number): string[] {
  if (!value) return [];
  const cards = parseBoardCardsText(value);
  if (cards.length !== expectedCount) return [];
  return cards;
}

function buildBoardCardsFromHash(params: URLSearchParams): string[] {
  const board = ["", "", "", "", ""];
  const flop = parseBoardHashParam(params.get("flop"), 3);
  const turn = parseBoardHashParam(params.get("turn"), 1);
  const river = parseBoardHashParam(params.get("river"), 1);
  for (let i = 0; i < flop.length; i += 1) board[i] = flop[i]!;
  if (turn[0]) board[3] = turn[0];
  if (river[0]) board[4] = river[0];
  return board;
}

function actionTokenFor(action: number, context: ActionContext): string {
  if (action === 0) return "f";
  if (action === 1) return "c";
  if (action === context.allInIndex) return "a";
  return `r${formatChips(displayAmountForRaise(action, context))}`;
}

function actionFromHashToken(
  token: string,
  legalActionsIn: readonly number[],
  context: ActionContext,
): number | undefined {
  const normalized = token.trim().toLowerCase();
  if (!normalized) return undefined;
  if ((normalized === "f" || normalized === "fold") && legalActionsIn.includes(0)) return 0;
  if (
    (normalized === "c" ||
      normalized === "call" ||
      normalized === "check" ||
      normalized === "x") &&
    legalActionsIn.includes(1)
  ) {
    return 1;
  }
  if (
    (normalized === "a" || normalized === "allin" || normalized === "all-in") &&
    legalActionsIn.includes(context.allInIndex)
  ) {
    return context.allInIndex;
  }
  if (normalized.startsWith("r") || normalized.startsWith("b")) {
    const value = normalized.slice(1);
    if (!value) return raiseActionOptions(legalActionsIn, context)[0];
    return closestRaiseAction(value, legalActionsIn, context);
  }
  return undefined;
}

function actionsFromHash(value: string | undefined): number[] {
  if (!value) return [];
  const env = createPublicEnv(LOCAL_ENV_DEFAULTS, {
    button: HERO_PLAYER,
    stack: String(LOCAL_ENV_DEFAULTS.stack),
    sb: String(LOCAL_ENV_DEFAULTS.sb),
    bb: String(LOCAL_ENV_DEFAULTS.bb),
  });
  const parsed: number[] = [];
  for (const token of value.split("-")) {
    const legal = env.legalBinsAmountAndMask();
    const action = actionFromHashToken(token, legalActionList(legal.mask), contextFromEnv(env, legal));
    if (action === undefined || !legal.mask[action]) break;
    parsed.push(action);
    env.stepBin(action, legal);
    if (env.done) break;
  }
  return parsed;
}

function actionTokensForActions(actionsIn: readonly number[]): string {
  const env = createPublicEnv(LOCAL_ENV_DEFAULTS, {
    button: HERO_PLAYER,
    stack: String(LOCAL_ENV_DEFAULTS.stack),
    sb: String(LOCAL_ENV_DEFAULTS.sb),
    bb: String(LOCAL_ENV_DEFAULTS.bb),
  });
  const tokens: string[] = [];
  for (const action of actionsIn) {
    const legal = env.legalBinsAmountAndMask();
    if (!legal.mask[action]) break;
    tokens.push(actionTokenFor(action, contextFromEnv(env, legal)));
    env.stepBin(action, legal);
    if (env.done) break;
  }
  return tokens.join("-");
}

function compactCardText(cards: readonly string[]): string {
  return cards.join("").replace(/\s+/g, "");
}

function parseHashInputs(hash: string): HashInputs {
  const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
  return {
    hand: params.get("hand") ?? undefined,
    actions: params.get("actions") ?? undefined,
    boardCards: buildBoardCardsFromHash(params),
  };
}

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = ["s", "h", "d", "c"] as const;
const SUIT_GLYPHS: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RANK_INDEX = new Map(RANKS.map((rank, index) => [rank, index]));

type HeroCards = [string, string];
type HandRangeKind = "pair" | "suited" | "offsuit";

interface RangeCell {
  key: string;
  label: string;
  kind: HandRangeKind;
}

interface ComboOption {
  cards: HeroCards;
  blocked: boolean;
}

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

function parseHeroHandText(input: string): HeroCards {
  const text = input.trim();
  if (!text) throw new Error("Enter both hero cards");

  const compact = text.replace(/[\s,]+/g, "");
  const compactMatch = /^([2-9TJQKA]|10)([cdhs])([2-9TJQKA]|10)([cdhs])$/i.exec(compact);
  const cards = compactMatch
    ? [
        formatCard(parseCard(`${compactMatch[1]}${compactMatch[2]}`)),
        formatCard(parseCard(`${compactMatch[3]}${compactMatch[4]}`)),
      ]
    : parseCards(text).map((card) => formatCard(card));

  if (cards.length !== 2) throw new Error("Enter exactly two hero cards");
  if (cards[0] === cards[1]) throw new Error(`Duplicate card ${cards[0]}`);
  return [cards[0]!, cards[1]!];
}

function parseBoardCardsText(input: string): string[] {
  const text = input.trim();
  if (!text) return [];
  if (/[\s,]+/.test(text)) {
    return parseCards(text).map((card) => formatCard(card));
  }

  const matches = Array.from(text.matchAll(/([2-9TJQKA]|10)([cdhs])/gi));
  const consumed = matches.map((match) => match[0]).join("");
  if (consumed.length !== text.length) {
    return parseCards(text).map((card) => formatCard(card));
  }
  return matches.map((match) => formatCard(parseCard(`${match[1]}${match[2]}`)));
}

function rankIndex(rank: string): number {
  return RANK_INDEX.get(rank) ?? Number.POSITIVE_INFINITY;
}

function orderedRanks(rankA: string, rankB: string): [string, string] {
  return rankIndex(rankA) <= rankIndex(rankB) ? [rankA, rankB] : [rankB, rankA];
}

function rangeKeyFromCards(cards: HeroCards): string {
  const rankA = cards[0][0] ?? "";
  const rankB = cards[1][0] ?? "";
  if (rankA === rankB) return `${rankA}${rankA}`;
  const [high, low] = orderedRanks(rankA, rankB);
  return `${high}${low}${cards[0][1] === cards[1][1] ? "s" : "o"}`;
}

function rangeCell(rowRank: string, colRank: string): RangeCell {
  if (rowRank === colRank) {
    return {
      key: `${rowRank}${colRank}`,
      label: `${rowRank}${colRank}`,
      kind: "pair",
    };
  }
  if (rankIndex(rowRank) < rankIndex(colRank)) {
    return {
      key: `${rowRank}${colRank}s`,
      label: `${rowRank}${colRank}s`,
      kind: "suited",
    };
  }
  return {
    key: `${colRank}${rowRank}o`,
    label: `${colRank}${rowRank}o`,
    kind: "offsuit",
  };
}

function rangeOptions(rangeKey: string, blockedCards: ReadonlySet<string>): ComboOption[] {
  const ranks = rangeKey.slice(0, 2);
  const modifier = rangeKey[2];
  const rankA = ranks[0] ?? "";
  const rankB = ranks[1] ?? "";
  const options: HeroCards[] = [];

  if (rankA === rankB) {
    for (let i = 0; i < SUITS.length; i += 1) {
      for (let j = i + 1; j < SUITS.length; j += 1) {
        options.push([`${rankA}${SUITS[i]}`, `${rankB}${SUITS[j]}`]);
      }
    }
  } else if (modifier === "s") {
    for (const suit of SUITS) {
      options.push([`${rankA}${suit}`, `${rankB}${suit}`]);
    }
  } else {
    for (const suitA of SUITS) {
      for (const suitB of SUITS) {
        if (suitA !== suitB) options.push([`${rankA}${suitA}`, `${rankB}${suitB}`]);
      }
    }
  }

  return options.map((cards) => ({
    cards,
    blocked: blockedCards.has(cards[0]) || blockedCards.has(cards[1]),
  }));
}

function comboMatchesHero(combo: HeroCards, hero?: HeroCards): boolean {
  if (!hero) return false;
  return combo.includes(hero[0]) && combo.includes(hero[1]);
}

function HeroHandSelector(props: {
  value: string;
  parsedCards?: HeroCards | undefined;
  error?: string | undefined;
  selectedRangeKey: string;
  blockedCards: ReadonlySet<string>;
  onTextChange: (value: string) => void;
  onNormalizeText: () => void;
  onRangeChange: (rangeKey: string) => void;
  onComboChange: (cards: HeroCards) => void;
}): JSX.Element {
  const [open, setOpen] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  let exactInput: HTMLInputElement | undefined;
  let comboContainer: HTMLDivElement | undefined;
  const selectedOptions = createMemo(() => rangeOptions(props.selectedRangeKey, props.blockedCards));
  const exactStateClass = createMemo(() => (props.error ? "invalid" : "valid"));

  function rangeIsBlocked(rangeKey: string): boolean {
    return rangeOptions(rangeKey, props.blockedCards).every((option) => option.blocked);
  }

  function editHand(): void {
    setEditing(true);
    queueMicrotask(() => {
      exactInput?.focus();
      exactInput?.select();
    });
  }

  function finishEditing(): void {
    props.onNormalizeText();
    setEditing(false);
  }

  function scrollCombosIntoView(): void {
    requestAnimationFrame(() => {
      const rect = comboContainer?.getBoundingClientRect();
      if (!rect) return;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
      if (rect.top >= 0 && rect.bottom <= viewportHeight) return;
      window.scrollBy({
        top: rect.bottom - viewportHeight + 12,
        behavior: "smooth",
      });
    });
  }

  return (
    <div class="hero-hand-selector">
      <div class="hero-hand-row">
        <Show
          when={editing()}
          fallback={
            <button
              type="button"
              class={`exact-hand-button ${props.error ? "invalid" : ""}`}
              onClick={editHand}
            >
              <Show
                when={props.parsedCards}
                fallback={<span class="exact-hand-raw">{props.value || "As Kd"}</span>}
              >
                {(cards) => (
                  <>
                    <CardChip value={cards()[0]} />
                    <CardChip value={cards()[1]} />
                  </>
                )}
              </Show>
            </button>
          }
        >
          <input
            ref={exactInput}
            class={`exact-hand-input ${exactStateClass()}`}
            value={props.value}
            placeholder="As Kd"
            onInput={(event) => props.onTextChange(event.currentTarget.value)}
            onBlur={finishEditing}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") setEditing(false);
            }}
          />
        </Show>
        <button
          type="button"
          class={`range-toggle ${open() ? "open" : ""}`}
          aria-expanded={open()}
          onClick={() => {
            if (editing()) finishEditing();
            setOpen((current) => !current);
          }}
        >
          <span>Grid</span>
          <ChevronDown size={16} />
        </button>
      </div>
      <Show when={props.error}>
        <span class="inline-error">{props.error}</span>
      </Show>
      <Show when={open()}>
        <div class="range-grid" role="grid" aria-label="Starting hand range grid">
          <For each={RANKS}>
            {(rowRank) => (
              <For each={RANKS}>
                {(colRank) => {
                  const cell = rangeCell(rowRank, colRank);
                  const blocked = () => rangeIsBlocked(cell.key);
                  return (
                    <button
                      type="button"
                      role="gridcell"
                      class={`range-cell ${cell.kind} ${props.selectedRangeKey === cell.key ? "selected" : ""}`}
                      disabled={blocked()}
                      onClick={() => {
                        props.onRangeChange(cell.key);
                        scrollCombosIntoView();
                      }}
                      title={cell.label}
                    >
                      {cell.label}
                    </button>
                  );
                }}
              </For>
            )}
          </For>
        </div>
        <div class="range-combo-head">
          <span>{selectedOptions().length} combos</span>
        </div>
        <div class="range-combos" ref={comboContainer}>
          <For each={selectedOptions()}>
            {(option) => (
              <button
                type="button"
                class={`range-combo ${comboMatchesHero(option.cards, props.parsedCards) ? "selected" : ""}`}
                disabled={option.blocked}
                onClick={() => {
                  props.onComboChange(option.cards);
                  setOpen(false);
                }}
              >
                <CardChip value={option.cards[0]} />
                <CardChip value={option.cards[1]} />
              </button>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}

function ActionInput(props: {
  actor: PlayerIndex;
  streetLabel?: string | undefined;
  legalActions: number[];
  context: ActionContext;
  onAction: (action: number) => void;
}): JSX.Element {
  const [editingRaise, setEditingRaise] = createSignal(false);
  const [raiseValue, setRaiseValue] = createSignal("");
  let raiseInput: HTMLInputElement | undefined;
  const raiseActions = createMemo(() =>
    props.legalActions.filter((action) => action >= 2 && action < props.context.allInIndex),
  );
  const directActions = createMemo(() =>
    props.legalActions.filter((action) => action < 2),
  );
  const allInAction = createMemo(() =>
    props.legalActions.includes(props.context.allInIndex) ? props.context.allInIndex : undefined,
  );
  const raiseLabel = createMemo(() => (props.context.toCall > 0 ? "Raise" : "Bet"));
  const raisePlaceholder = createMemo(() => {
    const first = raiseActions()[0];
    if (first === undefined) return "";
    const amount = props.context.amounts[first] ?? 0;
    return formatChips(props.context.toCall > 0 ? props.context.meCommitted + amount : amount);
  });

  function openRaiseInput(): void {
    setRaiseValue("");
    setEditingRaise(true);
    queueMicrotask(() => {
      raiseInput?.focus();
      raiseInput?.select();
    });
  }

  function commitRaise(): void {
    const target = Number(raiseValue());
    if (!Number.isFinite(target) || target <= 0) {
      setEditingRaise(false);
      return;
    }
    let bestAction = raiseActions()[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const action of raiseActions()) {
      const amount = props.context.amounts[action] ?? 0;
      const displayAmount =
        props.context.toCall > 0 ? props.context.meCommitted + amount : amount;
      const distance = Math.abs(displayAmount - target);
      if (distance < bestDistance) {
        bestAction = action;
        bestDistance = distance;
      }
    }
    setEditingRaise(false);
    if (bestAction !== undefined) props.onAction(bestAction);
  }

  return (
    <div class="action-input-row">
      <span class="street-marker">{props.streetLabel ?? ""}</span>
      <span class={`actor-pill ${props.actor === HERO_PLAYER ? "hero" : "villain"}`}>
        {playerLabel(props.actor)}
      </span>
      <div class="action-buttons">
        <For each={directActions()}>
          {(action) => (
            <button type="button" onClick={() => props.onAction(action)}>
              {formatActionLabel(action, props.context)}
            </button>
          )}
        </For>
        <Show when={raiseActions().length > 0}>
          <Show
            when={editingRaise()}
            fallback={
              <button type="button" onClick={openRaiseInput}>
                {raiseLabel()}
              </button>
            }
          >
            <input
              ref={raiseInput}
              class="raise-input"
              value={raiseValue()}
              inputmode="decimal"
              placeholder={raisePlaceholder()}
              onInput={(event) => setRaiseValue(event.currentTarget.value)}
              onBlur={() => {
                if (raiseValue()) commitRaise();
                else setEditingRaise(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") commitRaise();
                if (event.key === "Escape") setEditingRaise(false);
              }}
            />
          </Show>
        </Show>
        <Show when={allInAction()}>
          {(action) => (
            <button type="button" onClick={() => props.onAction(action())}>
              {formatActionLabel(action(), props.context)}
            </button>
          )}
        </Show>
      </div>
      <span class="pot-size">Pot {formatChips(props.context.pot)}</span>
    </div>
  );
}

function raiseActionOptions(
  legalActionsIn: readonly number[],
  context: ActionContext,
): number[] {
  return legalActionsIn.filter((action) => action >= 2 && action < context.allInIndex);
}

function displayAmountForRaise(action: number, context: ActionContext): number {
  const amount = context.amounts[action] ?? 0;
  return context.toCall > 0 ? context.meCommitted + amount : amount;
}

function closestRaiseAction(
  value: string,
  legalActionsIn: readonly number[],
  context: ActionContext,
): number | undefined {
  const target = Number(value);
  if (!Number.isFinite(target) || target <= 0) return undefined;
  let bestAction = raiseActionOptions(legalActionsIn, context)[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const action of raiseActionOptions(legalActionsIn, context)) {
    const distance = Math.abs(displayAmountForRaise(action, context) - target);
    if (distance < bestDistance) {
      bestAction = action;
      bestDistance = distance;
    }
  }
  return bestAction;
}

function ActionRowButtons(props: {
  action: number;
  legalActions: number[];
  context: ActionContext;
  onAction: (action: number) => void;
}): JSX.Element {
  const [editingRaise, setEditingRaise] = createSignal(false);
  const [raiseValue, setRaiseValue] = createSignal("");
  let raiseInput: HTMLInputElement | undefined;
  const raiseActions = createMemo(() => raiseActionOptions(props.legalActions, props.context));
  const isRaiseAction = createMemo(() => props.action >= 2 && props.action < props.context.allInIndex);
  const directActions = createMemo(() => props.legalActions.filter((action) => action < 2));
  const allInAction = createMemo(() =>
    props.legalActions.includes(props.context.allInIndex) ? props.context.allInIndex : undefined,
  );
  const raiseLabel = createMemo(() =>
    isRaiseAction()
      ? formatActionLabel(props.action, props.context)
      : props.context.toCall > 0 ? "Raise" : "Bet",
  );

  function openRaiseInput(): void {
    setRaiseValue(isRaiseAction() ? formatChips(displayAmountForRaise(props.action, props.context)) : "");
    setEditingRaise(true);
    queueMicrotask(() => {
      raiseInput?.focus();
      raiseInput?.select();
    });
  }

  function commitRaise(): void {
    const action = closestRaiseAction(raiseValue(), props.legalActions, props.context);
    setEditingRaise(false);
    if (action !== undefined) props.onAction(action);
  }

  function historyButtonClass(action: number): string {
    return `history-action-button ${action === props.action ? "selected" : "muted"}`;
  }

  function historyButtonLabel(action: number): string {
    return action === props.action
      ? formatActionLabel(action, props.context)
      : shortActionLabel(action, props.context);
  }

  return (
    <div class="action-buttons history-actions">
      <Show
        when={editingRaise()}
        fallback={
          <>
            <For each={directActions()}>
              {(action) => (
                <button
                  type="button"
                  class={historyButtonClass(action)}
                  onClick={() => props.onAction(action)}
                  title={formatActionLabel(action, props.context)}
                >
                  {historyButtonLabel(action)}
                </button>
              )}
            </For>
            <Show when={raiseActions().length > 0}>
              <button
                type="button"
                class={`history-action-button ${isRaiseAction() ? "selected" : "muted"}`}
                onClick={openRaiseInput}
                title={raiseLabel()}
              >
                {isRaiseAction()
                  ? formatActionLabel(props.action, props.context)
                  : shortActionLabel(raiseActions()[0]!, props.context)}
              </button>
            </Show>
            <Show when={allInAction()}>
              {(action) => (
                <button
                  type="button"
                  class={historyButtonClass(action())}
                  onClick={() => props.onAction(action())}
                  title={formatActionLabel(action(), props.context)}
                >
                  {historyButtonLabel(action())}
                </button>
              )}
            </Show>
          </>
        }
      >
        <input
          ref={raiseInput}
          class="raise-input action-raise-input"
          value={raiseValue()}
          inputmode="decimal"
          onInput={(event) => setRaiseValue(event.currentTarget.value)}
          onBlur={() => {
            if (raiseValue()) commitRaise();
            else setEditingRaise(false);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") commitRaise();
            if (event.key === "Escape") setEditingRaise(false);
          }}
        />
      </Show>
    </div>
  );
}

function NumberStepper(props: {
  value: string;
  presets: number[];
  min?: number;
  onChange: (value: string) => void;
}): JSX.Element {
  const current = createMemo(() => Number(props.value));
  const minValue = () => props.min ?? 1;

  function nudge(delta: number) {
    const next = Math.max(minValue(), Math.trunc((Number.isFinite(current()) ? current() : minValue()) + delta));
    props.onChange(String(next));
  }

  return (
    <div class="stepper">
      <button
        type="button"
        class="stepper-btn"
        onClick={() => nudge(-1)}
        title="Decrement"
      >
        −
      </button>
      <input
        class="stepper-input"
        value={props.value}
        inputmode="numeric"
        onInput={(event) => props.onChange(event.currentTarget.value)}
      />
      <button
        type="button"
        class="stepper-btn"
        onClick={() => nudge(1)}
        title="Increment"
      >
        +
      </button>
      <div class="stepper-presets">
        <For each={props.presets}>
          {(preset) => (
            <button
              type="button"
              class={`preset ${current() === preset ? "active" : ""}`}
              onClick={() => props.onChange(String(preset))}
            >
              {preset}
            </button>
          )}
        </For>
      </div>
    </div>
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

function BoardEntry(props: {
  street: number;
  count: number;
  cards: readonly string[];
  disabled: ReadonlySet<string>;
  onTextChange: (cards: string[]) => void;
}): JSX.Element {
  const priorCount = createMemo(() => STREET_CARD_COUNTS[Math.max(0, props.street - 1)] ?? 0);
  const editCount = createMemo(() => props.count - priorCount());
  const priorCards = createMemo(() => props.cards.slice(0, priorCount()).filter(Boolean));
  const editableCards = createMemo(() => props.cards.slice(priorCount(), props.count).filter(Boolean));
  const [text, setText] = createSignal(editableCards().join(" "));
  const [error, setError] = createSignal("");
  const [editing, setEditing] = createSignal(false);
  const [pickerOpen, setPickerOpen] = createSignal(false);
  const [focusedKey, setFocusedKey] = createSignal("");
  let inputRef: HTMLInputElement | undefined;

  createEffect(() => {
    setText(editableCards().join(" "));
  });

  createEffect(() => {
    const key = `${props.street}:${props.count}`;
    const needsBoard =
      editCount() > 0 && props.cards.slice(priorCount(), props.count).some((card) => !card);
    if (!needsBoard || focusedKey() === key) return;
    setFocusedKey(key);
    setEditing(true);
    queueMicrotask(() => {
      inputRef?.focus();
      inputRef?.select();
      setPickerOpen(true);
    });
  });

  const enteredCards = createMemo(() => {
    try {
      return parseBoardCardsText(text()).slice(0, editCount());
    } catch {
      return [];
    }
  });

  const blockedCards = createMemo(() => {
    const blocked = new Set(props.disabled);
    for (const card of enteredCards()) {
      if (card) blocked.add(card);
    }
    return blocked;
  });

  function commitText(): void {
    try {
      const parsed = parseBoardCardsText(text());
      if (parsed.length !== editCount()) {
        throw new Error(`Enter ${editCount()} board card${editCount() === 1 ? "" : "s"}`);
      }
      props.onTextChange([...props.cards.slice(0, priorCount()), ...parsed]);
      setText(parsed.join(" "));
      setError("");
      setEditing(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  }

  function editBoard(): void {
    setEditing(true);
    queueMicrotask(() => {
      inputRef?.focus();
      inputRef?.select();
      setPickerOpen(true);
    });
  }

  function selectCard(card: string): void {
    const current = enteredCards();
    if (blockedCards().has(card)) return;
    const next =
      current.length >= editCount()
        ? [...current.slice(0, Math.max(0, editCount() - 1)), card]
        : [...current, card];
    setText(next.join(" "));
    setError("");
    if (next.length === editCount()) {
      props.onTextChange([...props.cards.slice(0, priorCount()), ...next]);
      setEditing(false);
      setPickerOpen(false);
    }
  }

  return (
    <div class="board-entry-row">
      <span class="street-marker">{STREET_NAMES[props.street]}</span>
      <div class="board-entry">
        <div class="board-entry-main">
          <Show when={priorCards().length > 0}>
            <div class="board-context">
              <For each={priorCards()}>{(card) => <CardChip value={card} />}</For>
            </div>
          </Show>
          <Show
            when={editing()}
            fallback={
              <button
                type="button"
                class={`board-entry-button ${error() ? "invalid" : ""}`}
                onClick={editBoard}
              >
                <Show
                  when={editableCards().length > 0}
                  fallback={<span class="exact-hand-raw">{text() || (editCount() === 3 ? "As Kd Qh" : "Js")}</span>}
                >
                  <For each={editableCards()}>{(card) => <CardChip value={card} />}</For>
                </Show>
              </button>
            }
          >
            <input
              ref={inputRef}
              class={`board-text-input ${error() ? "invalid" : ""}`}
              value={text()}
              placeholder={editCount() === 3 ? "As Kd Qh" : "Js"}
              onInput={(event) => {
                setText(event.currentTarget.value);
                setError("");
              }}
              onFocus={() => setPickerOpen(true)}
              onClick={() => setPickerOpen(true)}
              onBlur={() => {
                commitText();
                setPickerOpen(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  commitText();
                  event.currentTarget.blur();
                }
                if (event.key === "Escape") {
                  setEditing(false);
                  setPickerOpen(false);
                }
              }}
            />
          </Show>
        </div>
        <Show when={editing() && pickerOpen()}>
          <div
            class="board-card-picker"
            onMouseDown={(event) => event.preventDefault()}
          >
            <For each={SUITS}>
              {(suit) => (
                <For each={RANKS}>
                  {(rank) => {
                    const card = `${rank}${suit}`;
                    const taken = () => blockedCards().has(card);
                    return (
                      <button
                        type="button"
                        class={"card-cell " + suitClass(suit) + (taken() ? " taken" : "")}
                        disabled={taken()}
                        onClick={() => selectCard(card)}
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
        </Show>
        <Show when={error()}>
          <span class="inline-error">{error()}</span>
        </Show>
      </div>
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
  const [button, setButton] = createSignal<PlayerIndex>(HERO_PLAYER);
  const [stack, setStack] = createSignal("400");
  const [sb, setSb] = createSignal("1");
  const [bb, setBb] = createSignal("2");
  const [iterations, setIterations] = createSignal("16");
  const [depth, setDepth] = createSignal("6");
  const [heroHandText, setHeroHandText] = createSignal("As Kd");
  const [selectedRangeKey, setSelectedRangeKey] = createSignal("AKo");
  const [boardCards, setBoardCards] = createSignal(["", "", "", "", ""]);
  const [actions, setActions] = createSignal<number[]>([]);
  const [solveStatus, setSolveStatus] = createSignal("");
  const [solveError, setSolveError] = createSignal("");
  const [solveResult, setSolveResult] = createSignal<SolveResult>();
  const [hashHydrated, setHashHydrated] = createSignal(false);
  let applyingHash = false;
  let lastWrittenHash = "";

  function applyHashFromLocation(): void {
    applyingHash = true;
    try {
      const input = parseHashInputs(window.location.hash);
      if (input.hand) {
        const cards = parseHeroHandText(input.hand);
        setHeroHandText(`${cards[0]} ${cards[1]}`);
        setSelectedRangeKey(rangeKeyFromCards(cards));
      }
      setBoardCards(input.boardCards);
      setActions(actionsFromHash(input.actions));
      setSolveResult(undefined);
      setSolveError("");
      setSolveStatus("");
    } catch (error) {
      setSolveError(error instanceof Error ? error.message : String(error));
    } finally {
      applyingHash = false;
      setHashHydrated(true);
    }
  }

  onMount(() => {
    applyHashFromLocation();
    window.addEventListener("hashchange", applyHashFromLocation);
    void loadRuntime();
  });

  onCleanup(() => {
    window.removeEventListener("hashchange", applyHashFromLocation);
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
      const model = BetterFfnWebGpuModel.fromBuffers(device, manifest, loaded.weights);
      const evaluator = createBrowserCfrEvaluator(device, model);
      setRuntime({
        device,
        model,
        evaluator,
        manifest,
        cached: loaded.cached,
        usingSubgroups: device.features.has("subgroups" as GPUFeatureName),
      });
      setModelProgress({
        phase: loaded.cached ? "cache-hit" : "stored",
        message: loaded.cached ? "Model loaded from IndexedDB" : "Model loaded and cached",
      });
    } catch (error) {
      setModelError(error instanceof Error ? error.message : String(error));
    }
  }

  const parsedHeroHand = createMemo<{ cards: HeroCards } | { error: string }>(() => {
    try {
      return { cards: parseHeroHandText(heroHandText()) };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  });
  const parsedHeroCards = createMemo(() => {
    const hero = parsedHeroHand();
    return "cards" in hero ? hero.cards : undefined;
  });
  const heroHandError = createMemo(() => {
    const hero = parsedHeroHand();
    return "error" in hero ? hero.error : "";
  });

  const descriptor = createMemo<StateDescriptor | { error: string }>(() => {
    const hero = parsedHeroHand();
    try {
      const heroHand =
        "cards" in hero
          ? ([
              CARD_OPTIONS.includes(hero.cards[0]) ? cardFromOption(hero.cards[0]) : -1,
              CARD_OPTIONS.includes(hero.cards[1]) ? cardFromOption(hero.cards[1]) : -1,
            ] as [number, number])
          : undefined;
      const env = createPublicEnv(LOCAL_ENV_DEFAULTS, {
        button: button(),
        stack: stack(),
        sb: sb(),
        bb: bb(),
        ...(heroHand ? { forceDeck: buildForceDeck(heroHand, boardCards()) } : {}),
      });
      if (heroHand) {
        env.configureKnownCards({
          heroPlayer: HERO_PLAYER,
          heroHand,
        });
      }

      const rows: ActionRow[] = [];
      for (const action of actions()) {
        const legal = env.legalBinsAmountAndMask();
        rows.push({
          action,
          actor: env.toAct,
          legalMask: legal.mask,
          context: contextFromEnv(env, legal),
          street: env.street,
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
        finalStreet: env.street,
      };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  });

  const inferredStreet = createMemo(() => {
    const state = descriptor();
    return "error" in state ? 0 : Math.min(3, Math.max(0, state.finalStreet));
  });
  const inferredPublicCount = createMemo(() => STREET_CARD_COUNTS[inferredStreet()] ?? 0);
  const visibleBoardComplete = createMemo(() => {
    const publicCount = inferredPublicCount();
    return publicCount === 0 || boardCards().slice(0, publicCount).every(Boolean);
  });

  const parsedInputs = createMemo(() => {
    try {
      const hero = parsedHeroHand();
      if ("error" in hero) throw new Error(hero.error);
      const publicCount = inferredPublicCount();
      const visibleBoard = boardCards().slice(0, publicCount);
      if (visibleBoard.some((card) => card === "")) {
        const streetName = STREET_NAMES[inferredStreet()] ?? "current street";
        throw new Error(`Select every ${streetName.toLowerCase()} board card`);
      }
      const heroHand = [
        CARD_OPTIONS.includes(hero.cards[0]) ? cardFromOption(hero.cards[0]) : -1,
        CARD_OPTIONS.includes(hero.cards[1]) ? cardFromOption(hero.cards[1]) : -1,
      ] as [number, number];
      const publicCards = visibleBoard.map((card) => cardFromOption(card));
      const seen = new Set<number>();
      for (const card of [...heroHand, ...publicCards]) {
        if (card < 0) throw new Error("Invalid card selection");
        if (seen.has(card)) throw new Error(`Duplicate card ${formatCard(card)}`);
        seen.add(card);
      }
      return {
        heroHand,
        publicCards,
        forceDeck: buildForceDeck(heroHand, visibleBoard),
      };
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  });

  const usedCards = createMemo<Set<string>>(() => {
    const used = new Set<string>();
    const hero = parsedHeroHand();
    if ("cards" in hero) {
      for (const card of hero.cards) {
        used.add(card);
      }
    }
    const publicCount = inferredPublicCount();
    for (let i = 0; i < publicCount; i += 1) {
      const card = boardCards()[i];
      if (card) used.add(card);
    }
    return used;
  });

  const boardUsedCards = createMemo<Set<string>>(() => {
    const used = new Set<string>();
    const publicCount = inferredPublicCount();
    for (let i = 0; i < publicCount; i += 1) {
      const card = boardCards()[i];
      if (card) used.add(card);
    }
    return used;
  });

  const descriptorError = createMemo(() => {
    const current = descriptor();
    return "error" in current ? current.error : "";
  });
  const serializedHash = createMemo(() => {
    const params = new URLSearchParams();
    try {
      params.set("hand", compactCardText(parseHeroHandText(heroHandText())));
    } catch {
      params.set("hand", heroHandText().replace(/[\s,]+/g, ""));
    }
    params.set("actions", actionTokensForActions(actions()));
    const board = boardCards();
    params.set("flop", board.slice(0, 3).every(Boolean) ? compactCardText(board.slice(0, 3)) : "");
    params.set("turn", board[3] ? compactCardText([board[3]]) : "");
    params.set("river", board[4] ? compactCardText([board[4]]) : "");
    return `#${params.toString()}`;
  });
  createEffect(() => {
    const nextHash = serializedHash();
    if (!hashHydrated() || applyingHash) return;
    if (nextHash === lastWrittenHash || nextHash === window.location.hash) return;
    lastWrittenHash = nextHash;
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}${nextHash}`);
  });
  const solveInputError = createMemo(() => {
    const descriptorMessage = descriptorError();
    if (descriptorMessage) return descriptorMessage;
    const inputs = parsedInputs();
    return "error" in inputs ? inputs.error : "";
  });
  const modelProgressPercent = createMemo(() => {
    const progress = modelProgress();
    if (!progress.totalBytes || progress.loadedBytes === undefined) return undefined;
    return Math.min(100, (progress.loadedBytes / progress.totalBytes) * 100);
  });

  function nextAvailableCard(used: Set<number>): number {
    for (let card = 0; card < CARD_OPTIONS.length; card += 1) {
      if (!used.has(card)) {
        used.add(card);
        return card;
      }
    }
    throw new Error("No available card remains for force deck");
  }

  function buildForceDeck(
    heroHand: readonly [number, number],
    boardValues: readonly string[],
  ): number[] {
    const used = new Set<number>();
    for (const card of heroHand) {
      if (used.has(card)) throw new Error(`Duplicate card ${formatCard(card)}`);
      used.add(card);
    }

    const board = new Array<number>(5).fill(-1);
    for (let i = 0; i < Math.min(5, boardValues.length); i += 1) {
      const value = boardValues[i];
      if (!value) continue;
      const card = cardFromOption(value);
      if (used.has(card)) throw new Error(`Duplicate card ${formatCard(card)}`);
      used.add(card);
      board[i] = card;
    }

    const villainHand = [nextAvailableCard(used), nextAvailableCard(used)];
    for (let i = 0; i < board.length; i += 1) {
      if (board[i] === -1) board[i] = nextAvailableCard(used);
    }
    return [heroHand[0], heroHand[1], villainHand[0]!, villainHand[1]!, ...board];
  }

  function updateHeroHandText(value: string): void {
    setHeroHandText(value);
    try {
      setSelectedRangeKey(rangeKeyFromCards(parseHeroHandText(value)));
    } catch {
      // Keep the previously selected range visible while the user is typing.
    }
    setSolveResult(undefined);
  }

  function normalizeHeroHandText(): void {
    try {
      const cards = parseHeroHandText(heroHandText());
      setHeroHandText(`${cards[0]} ${cards[1]}`);
      setSelectedRangeKey(rangeKeyFromCards(cards));
    } catch {
      // Leave invalid text intact so the inline validation message stays actionable.
    }
  }

  function selectHeroCombo(cards: HeroCards): void {
    setHeroHandText(`${cards[0]} ${cards[1]}`);
    setSelectedRangeKey(rangeKeyFromCards(cards));
    setSolveResult(undefined);
  }

  function updateBoardCards(values: readonly string[]): void {
    setBoardCards((cards) => {
      const next = [...cards];
      for (let i = 0; i < Math.min(values.length, next.length); i += 1) {
        next[i] = values[i] ?? "";
      }
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
      const solveDepth = Math.max(2, Math.trunc(positiveNumber(depth(), 6)));
      const solveCfrAvg = false;
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
          stack: positiveNumber(stack(), LOCAL_ENV_DEFAULTS.stack),
          sb: positiveNumber(sb(), LOCAL_ENV_DEFAULTS.sb),
          bb: positiveNumber(bb(), LOCAL_ENV_DEFAULTS.bb),
          betBins: LOCAL_ENV_DEFAULTS.betBins,
          flopShowdown: LOCAL_ENV_DEFAULTS.flopShowdown,
          forceDeck: cards.forceDeck,
        },
        heroPlayer: HERO_PLAYER,
        heroHand: cards.heroHand,
        initialBeliefs: buildHeroOnlyBeliefs({
          heroPlayer: HERO_PLAYER,
          heroHand: cards.heroHand,
          publicCards: cards.publicCards,
        }),
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
        villainSummary: summarizeVillainRange(result.beliefsAtSpot, HERO_PLAYER),
      });
      setSolveStatus("Solve complete");
    } catch (error) {
      setSolveError(error instanceof Error ? error.message : String(error));
      setSolveStatus("");
    }
  }

  function legalActions(mask: readonly number[]): number[] {
    const actionsOut: number[] = [];
    for (let i = 0; i < mask.length; i += 1) {
      if (mask[i]) actionsOut.push(i);
    }
    return actionsOut;
  }

  function streetMarkerForRow(rows: readonly ActionRow[], index: number): string | undefined {
    const row = rows[index];
    if (!row) return undefined;
    if (row.street > 0) return undefined;
    if (index === 0 || row.street !== rows[index - 1]?.street) {
      return STREET_NAMES[row.street] ?? undefined;
    }
    return undefined;
  }

  function shouldShowBoardBeforeRow(rows: readonly ActionRow[], index: number): boolean {
    const row = rows[index];
    if (!row || row.street <= 0) return false;
    return index === 0 || row.street !== rows[index - 1]?.street;
  }

  function shouldShowBoardBeforeFinalAction(state: StateDescriptor): boolean {
    if (state.finalStreet <= 0) return false;
    return state.rows.every((row) => row.street !== state.finalStreet);
  }

  function finalStreetMarker(rows: readonly ActionRow[], finalStreet: number): string | undefined {
    const last = rows[rows.length - 1];
    if (!last || last.street !== finalStreet) return STREET_NAMES[finalStreet] ?? undefined;
    return undefined;
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
          <Show when={runtime()}>
            {(current) => (
              <span
                class={`feature-badge ${current().usingSubgroups ? "on" : "off"}`}
                title={
                  current().usingSubgroups
                    ? "BetterFFN subgroup kernels are enabled"
                    : "Using non-subgroup fallback kernels"
                }
              >
                Subgroups {current().usingSubgroups ? "on" : "off"}
              </span>
            )}
          </Show>
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
                <span>Button / small blind</span>
                <select
                  value={String(button())}
                  onChange={(event) => {
                    setButton(asPlayer(event.currentTarget.value));
                    setActions([]);
                    setSolveResult(undefined);
                  }}
                >
                  <option value="0">Hero</option>
                  <option value="1">Villain</option>
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
            </div>
            <HeroHandSelector
              value={heroHandText()}
              parsedCards={parsedHeroCards()}
              error={heroHandError()}
              selectedRangeKey={selectedRangeKey()}
              blockedCards={boardUsedCards()}
              onTextChange={updateHeroHandText}
              onNormalizeText={normalizeHeroHandText}
              onRangeChange={(rangeKey) => {
                setSelectedRangeKey(rangeKey);
                setSolveResult(undefined);
              }}
              onComboChange={selectHeroCombo}
            />
          </div>

          <div class="subgroup">
            <div class="subgroup-head">
              <span class="subgroup-title">Action sequence</span>
              <Show when={actions().length > 0}>
                <button
                  type="button"
                  class="link-button"
                  onClick={() => {
                    setActions(actions().slice(0, -1));
                    setSolveResult(undefined);
                  }}
                >
                  Undo last
                </button>
              </Show>
            </div>
            <Show when={!descriptorError()} fallback={<p class="inline-error">{descriptorError()}</p>}>
              <For each={(descriptor() as StateDescriptor).rows}>
                {(row, index) => (
                  <>
                    <Show when={shouldShowBoardBeforeRow((descriptor() as StateDescriptor).rows, index())}>
                      <BoardEntry
                        street={row.street}
                        count={STREET_CARD_COUNTS[row.street] ?? 0}
                        cards={boardCards()}
                        disabled={usedCards()}
                        onTextChange={updateBoardCards}
                      />
                    </Show>
                    <div class="action-row">
                      <span class="street-marker">{streetMarkerForRow((descriptor() as StateDescriptor).rows, index()) ?? ""}</span>
                      <span class={`actor-pill ${row.actor === HERO_PLAYER ? "hero" : "villain"}`}>
                        {playerLabel(row.actor)}
                      </span>
                      <ActionRowButtons
                        action={row.action}
                        legalActions={legalActions(row.legalMask)}
                        context={row.context}
                        onAction={(action) => setActionAt(index(), action)}
                      />
                      <button
                        type="button"
                        class="icon-button"
                        title="Remove action"
                        onClick={() => removeAction(index())}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </>
                )}
              </For>
              <Show when={shouldShowBoardBeforeFinalAction(descriptor() as StateDescriptor)}>
                <BoardEntry
                  street={inferredStreet()}
                  count={inferredPublicCount()}
                  cards={boardCards()}
                  disabled={usedCards()}
                  onTextChange={updateBoardCards}
                />
              </Show>
              <Show
                when={
                  visibleBoardComplete() &&
                  legalActions((descriptor() as StateDescriptor).finalLegalMask).length > 0
                }
              >
                <ActionInput
                  actor={(descriptor() as StateDescriptor).finalActor}
                  streetLabel={
                    inferredPublicCount() > 0
                      ? undefined
                      : finalStreetMarker(
                          (descriptor() as StateDescriptor).rows,
                          (descriptor() as StateDescriptor).finalStreet,
                        )
                  }
                  legalActions={legalActions((descriptor() as StateDescriptor).finalLegalMask)}
                  context={(descriptor() as StateDescriptor).finalContext}
                  onAction={addAction}
                />
              </Show>
            </Show>
          </div>

          <div class="subgroup">
            <span class="subgroup-title">Solve</span>
            <div class="stepper-row">
              <span class="stepper-label">Iterations</span>
              <NumberStepper
                value={iterations()}
                presets={[8, 32, 128, 512, 2048]}
                min={1}
                onChange={(value) => {
                  setIterations(value);
                  setSolveResult(undefined);
                }}
              />
            </div>
            <div class="stepper-row">
              <span class="stepper-label">Depth</span>
              <NumberStepper
                value={depth()}
                presets={[4, 5, 6, 7]}
                min={2}
                onChange={(value) => {
                  setDepth(value);
                  setSolveResult(undefined);
                }}
              />
            </div>
            <Show when={solveInputError() && !descriptorError()}>
              <p class="inline-error">{solveInputError()}</p>
            </Show>
            <button
              type="button"
              class="solve-button"
              disabled={!runtime() || Boolean(solveInputError())}
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
                  <span>{runtime()?.usingSubgroups ? "subgroups on" : "subgroups off"}</span>
                  <span>actor {playerLabel(solved().result.actor)}</span>
                </div>

                <section class="subsection no-border">
                  <h3>Aggregated strategy</h3>
                  <p class="subtle">Probability of each action across the actor's range</p>
                  <StrategyTable
                    labels={solved().result.actionLabels}
                    legalMask={solved().result.legalMask}
                    actionProbs={solved().result.actionProbs}
                    context={(descriptor() as StateDescriptor).finalContext}
                  />
                </section>

                <HeroPolicy
                  labels={solved().result.actionLabels}
                  legalMask={solved().result.legalMask}
                  policy={solved().result.policy}
                  handIndex={solved().heroHandIndex}
                  actor={solved().result.actor}
                  heroPlayer={HERO_PLAYER}
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
    <div class="strategy-bars">
      <For each={props.labels}>
        {(label, index) => {
          const legal = props.legalMask[index()] === 1;
          const value = props.actionProbs[index()] ?? 0;
          if (!legal && value <= 0) return null;
          const name = props.context ? formatActionLabel(index(), props.context) : label;
          return (
            <div class={`bar-row ${legal ? "" : "muted"}`}>
              <span class="bar-label">{name}</span>
              <div class="bar-track">
                <div class="bar-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
              </div>
              <span class="bar-value">{formatPercent(value)}</span>
            </div>
          );
        }}
      </For>
    </div>
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
      <h3>Hero hand policy</h3>
      <p class="subtle">
        {props.actor === props.heroPlayer
          ? `Action distribution for this exact hand`
          : `Villain (P${props.actor}) is to act; this is hero's hypothetical response`}
      </p>
      <div class="strategy-bars">
        <For each={row()}>
          {(item) => {
            if (!item.legal && item.value <= 0) return null;
            return (
              <div class={`bar-row ${item.legal ? "" : "muted"}`}>
                <span class="bar-label">{item.label}</span>
                <div class="bar-track">
                  <div class="bar-fill alt" style={{ width: `${Math.min(100, item.value * 100)}%` }} />
                </div>
                <span class="bar-value">{formatPercent(item.value)}</span>
              </div>
            );
          }}
        </For>
      </div>
    </section>
  );
}

function RangeSummaryView(props: { summary: RangeSummary }): JSX.Element {
  return (
    <section class="subsection">
      <h3>Villain range</h3>
      <div class="metadata">
        <span>{props.summary.combos} combos</span>
        <span>mass {props.summary.mass.toFixed(4)}</span>
      </div>
      <div class="range-list">
        <For each={props.summary.top}>
          {(item) => {
            const cards = item.hand.split(" ");
            return (
              <div class="range-row">
                <span class="range-cards">
                  <CardChip value={cards[0] ?? ""} />
                  <CardChip value={cards[1] ?? ""} />
                </span>
                <strong>{formatPercent(item.weight)}</strong>
              </div>
            );
          }}
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
