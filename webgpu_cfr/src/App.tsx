import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import type { JSX } from "solid-js";
import {
  AlertTriangle,
  ChevronDown,
  Cpu,
  Play,
  RotateCcw,
  Trash2,
  X,
} from "lucide-solid";
import type { ModelCacheProgress } from "./modelCache.js";
import {
  parseCard,
  parseCards,
  formatCard,
  handComboIndex,
  handComboCards,
} from "./cards.js";
import { buildPublicBeliefs } from "./beliefs.js";
import { PublicHunlEnv, NUM_HANDS, DEFAULT_FORCE_DECK, type LegalBins } from "./hunlEnv.js";
import type {
  BetterFfnManifest,
  BrowserEvaluationResult,
  PlayerIndex,
} from "./types.js";
import type {
  SolverEvaluateSpotRequest,
  SolverWorkerRequest,
  SolverWorkerResponse,
} from "./solverWorkerMessages.js";

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

function waitForBrowserPaint(): Promise<void> {
  if (typeof requestAnimationFrame === "undefined") return Promise.resolve();
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

interface Runtime {
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
  button: PlayerIndex;
  stack: string;
  sb: string;
  bb: string;
}

interface SolveResult {
  result: BrowserEvaluationResult;
  elapsedMs: number;
  iterations: number;
  depth: number;
  cfrAvg: boolean;
  heroHandIndex: number;
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

function defaultNumberText(value: number): string {
  return String(value);
}

function hashNumberParam(
  params: URLSearchParams,
  key: string,
  fallback: number,
): string {
  const value = params.get(key);
  if (value === null) return defaultNumberText(fallback);
  const parsed = positiveNumber(value, fallback);
  return parsed === fallback ? defaultNumberText(fallback) : value.trim();
}

function hashHeroParam(value: string | null): PlayerIndex {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "bb" || normalized === "1") return 1;
  return HERO_PLAYER;
}

function setHashNumberParam(
  params: URLSearchParams,
  key: string,
  value: string,
  fallback: number,
): void {
  const parsed = positiveNumber(value, fallback);
  if (parsed !== fallback) params.set(key, value.trim());
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

function actionsFromHash(
  value: string | undefined,
  options: Pick<HashInputs, "button" | "stack" | "sb" | "bb">,
): number[] {
  if (!value) return [];
  const env = createPublicEnv(LOCAL_ENV_DEFAULTS, {
    button: options.button,
    stack: options.stack,
    sb: options.sb,
    bb: options.bb,
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

function actionTokensForActions(
  actionsIn: readonly number[],
  options: {
    button: PlayerIndex;
    stack: string;
    sb: string;
    bb: string;
  },
): string {
  const env = createPublicEnv(LOCAL_ENV_DEFAULTS, {
    button: options.button,
    stack: options.stack,
    sb: options.sb,
    bb: options.bb,
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
    button: hashHeroParam(params.get("hero")),
    stack: hashNumberParam(params, "stack", LOCAL_ENV_DEFAULTS.stack),
    sb: hashNumberParam(params, "sb", LOCAL_ENV_DEFAULTS.sb),
    bb: hashNumberParam(params, "bb", LOCAL_ENV_DEFAULTS.bb),
  };
}

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = ["s", "h", "d", "c"] as const;
const SUIT_GLYPHS: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RANK_INDEX = new Map(RANKS.map((rank, index) => [rank, index]));
const UNBLOCKED_CARDS: ReadonlySet<string> = new Set<string>();

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

interface RangeGridCombo {
  hand: string;
  weight: number;
}

interface RangeGridCell extends RangeCell {
  total: number;
  alpha: number;
  title: string;
}

interface RangeGridModel {
  label: string;
  rows: RangeGridCell[][];
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
  const [webGpuError, setWebGpuError] = createSignal("");
  const [button, setButton] = createSignal<PlayerIndex>(HERO_PLAYER);
  const [stack, setStack] = createSignal("400");
  const [sb, setSb] = createSignal("1");
  const [bb, setBb] = createSignal("2");
  const [iterations, setIterations] = createSignal("400");
  const [depth, setDepth] = createSignal("5");
  const [heroHandText, setHeroHandText] = createSignal("As Kd");
  const [selectedRangeKey, setSelectedRangeKey] = createSignal("AKo");
  const [boardCards, setBoardCards] = createSignal(["", "", "", "", ""]);
  const [actions, setActions] = createSignal<number[]>([]);
  const [solveError, setSolveError] = createSignal("");
  const [solveResult, setSolveResult] = createSignal<SolveResult>();
  const [isSolving, setIsSolving] = createSignal(false);
  const [solveProgress, setSolveProgress] = createSignal<number | undefined>();
  const [hashHydrated, setHashHydrated] = createSignal(false);
  let solverWorker: Worker | undefined;
  let nextSolveId = 0;
  let activeSolve:
    | {
        id: number;
        depth: number;
        resolve: (result: BrowserEvaluationResult) => void;
        reject: (error: Error) => void;
      }
    | undefined;
  let applyingHash = false;

  function clearSolveOutput(): void {
    setSolveResult(undefined);
    setSolveProgress(undefined);
    setSolveError("");
  }

  function reportWebGpuError(message: string): void {
    setWebGpuError(message);
    if (isSolving()) {
      setSolveError(message);
      setSolveProgress(undefined);
      setIsSolving(false);
      activeSolve?.reject(new Error(message));
      activeSolve = undefined;
    }
  }

  function rejectActiveSolve(message: string): void {
    activeSolve?.reject(new Error(message));
    activeSolve = undefined;
    if (isSolving()) {
      setSolveError(message);
      setSolveProgress(undefined);
      setIsSolving(false);
    }
  }

  function handleWorkerMessage(message: SolverWorkerResponse): void {
    if (message.type === "model-progress") {
      setModelProgress(message.progress);
    } else if (message.type === "ready") {
      setRuntime(message.runtime);
    } else if (message.type === "webgpu-error") {
      reportWebGpuError(message.message);
    } else if (message.type === "error") {
      setModelError(message.message);
      rejectActiveSolve(message.message);
    } else if (message.type === "solve-progress") {
      if (activeSolve?.id !== message.id) return;
      setSolveProgress(message.progress.percent);
    } else if (message.type === "solve-result") {
      if (activeSolve?.id !== message.id) return;
      activeSolve.resolve(message.result);
      activeSolve = undefined;
    } else {
      if (activeSolve?.id !== message.id) return;
      activeSolve.reject(new Error(message.message));
      activeSolve = undefined;
    }
  }

  function ensureSolverWorker(): Worker {
    if (solverWorker) return solverWorker;
    const worker = new Worker(new URL("./solverWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.addEventListener("message", (event: MessageEvent<SolverWorkerResponse>) => {
      handleWorkerMessage(event.data);
    });
    worker.addEventListener("error", (event) => {
      const message = event.message || "Solver worker failed";
      setModelError(message);
      rejectActiveSolve(message);
    });
    worker.addEventListener("messageerror", () => {
      const message = "Solver worker sent an unreadable message";
      setModelError(message);
      rejectActiveSolve(message);
    });
    solverWorker = worker;
    return worker;
  }

  function postWorkerMessage(message: SolverWorkerRequest): void {
    ensureSolverWorker().postMessage(message);
  }

  function evaluateSpotInWorker(
    request: SolverEvaluateSpotRequest,
    solveDepth: number,
  ): Promise<BrowserEvaluationResult> {
    if (!runtime()) throw new Error("solver worker is not ready");
    if (activeSolve) throw new Error("a solve is already running");
    const id = nextSolveId + 1;
    nextSolveId = id;
    return new Promise((resolve, reject) => {
      activeSolve = { id, depth: solveDepth, resolve, reject };
      postWorkerMessage({ type: "solve", id, request });
    });
  }

  function applyHashFromLocation(): void {
    applyingHash = true;
    try {
      const input = parseHashInputs(window.location.hash);
      if (input.hand) {
        const cards = parseHeroHandText(input.hand);
        setHeroHandText(`${cards[0]} ${cards[1]}`);
        setSelectedRangeKey(rangeKeyFromCards(cards));
      }
      setButton(input.button);
      setStack(input.stack);
      setSb(input.sb);
      setBb(input.bb);
      setBoardCards(input.boardCards);
      setActions(actionsFromHash(input.actions, input));
      clearSolveOutput();
      setSolveError("");
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
    solverWorker?.postMessage({ type: "dispose" } satisfies SolverWorkerRequest);
    solverWorker?.terminate();
    solverWorker = undefined;
  });

  function loadRuntime(): void {
    try {
      setModelProgress({ phase: "manifest", message: "Starting solver worker" });
      postWorkerMessage({ type: "init", manifestUrl: MODEL_MANIFEST_URL });
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
    if (button() !== HERO_PLAYER) params.set("hero", "bb");
    setHashNumberParam(params, "stack", stack(), LOCAL_ENV_DEFAULTS.stack);
    setHashNumberParam(params, "sb", sb(), LOCAL_ENV_DEFAULTS.sb);
    setHashNumberParam(params, "bb", bb(), LOCAL_ENV_DEFAULTS.bb);
    const actionTokens = actionTokensForActions(actions(), {
      button: button(),
      stack: stack(),
      sb: sb(),
      bb: bb(),
    });
    if (actionTokens) params.set("actions", actionTokens);
    const board = boardCards();
    if (board.slice(0, 3).every(Boolean)) {
      params.set("flop", compactCardText(board.slice(0, 3)));
    }
    if (board[3]) params.set("turn", compactCardText([board[3]]));
    if (board[4]) params.set("river", compactCardText([board[4]]));
    return `#${params.toString()}`;
  });
  createEffect(() => {
    const nextHash = serializedHash();
    if (!hashHydrated() || applyingHash) return;
    if (nextHash === window.location.hash) return;
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
    clearSolveOutput();
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
    clearSolveOutput();
  }

  function updateBoardCards(values: readonly string[]): void {
    setBoardCards((cards) => {
      const next = [...cards];
      for (let i = 0; i < Math.min(values.length, next.length); i += 1) {
        next[i] = values[i] ?? "";
      }
      return next;
    });
    clearSolveOutput();
  }

  function setActionAt(index: number, action: number): void {
    setActions((current) => current.map((value, i) => (i === index ? action : value)));
    clearSolveOutput();
  }

  function removeAction(index: number): void {
    setActions((current) => current.filter((_, i) => i !== index));
    clearSolveOutput();
  }

  function addAction(action: number): void {
    setActions((current) => [...current, action]);
    clearSolveOutput();
  }

  async function solve(): Promise<void> {
    const current = runtime();
    const cards = parsedInputs();
    const state = descriptor();
    if (!current || "error" in cards || "error" in state) return;

    try {
      setSolveError("");
      setWebGpuError("");
      const solveIterations = Math.max(1, Math.trunc(positiveNumber(iterations(), 400)));
      const solveDepth = Math.max(2, Math.trunc(positiveNumber(depth(), 5)));
      const solveCfrAvg = false;
      setIsSolving(true);
      setSolveProgress(0);
      setSolveResult(undefined);
      const request: SolverEvaluateSpotRequest = {
        spot: [...actions()],
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
          ...(cards.publicCards.length > 0 ? { forceDeck: cards.forceDeck } : {}),
        },
        heroPlayer: HERO_PLAYER,
        heroHand: cards.heroHand,
        initialBeliefs: buildPublicBeliefs({ publicCards: cards.publicCards }),
      };
      await waitForBrowserPaint();
      const started = performance.now();
      const result = await evaluateSpotInWorker(request, solveDepth);
      const elapsedMs = performance.now() - started;
      const heroHandIndex = handComboIndex(cards.heroHand[0], cards.heroHand[1]);
      setSolveResult({
        result,
        elapsedMs,
        iterations: solveIterations,
        depth: solveDepth,
        cfrAvg: solveCfrAvg,
        heroHandIndex,
      });
      setSolveProgress(100);
    } catch (error) {
      setSolveError(error instanceof Error ? error.message : String(error));
      setSolveProgress(undefined);
    } finally {
      setIsSolving(false);
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
          <h1>holdem.computer</h1>
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
      <Show when={webGpuError()}>
        <div class="notice error">
          <AlertTriangle size={18} />
          <span>{webGpuError()}</span>
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
                clearSolveOutput();
              }}
            >
              <RotateCcw size={16} />
            </button>
          </div>

          <div class="subgroup">
            <span class="subgroup-title">Game</span>
            <div class="game-config-row">
              <label class="field">
                <span>Hero</span>
                <select
                  value={String(button())}
                  onChange={(event) => {
                    setButton(asPlayer(event.currentTarget.value));
                    setActions([]);
                    clearSolveOutput();
                  }}
                >
                  <option value="0">SB</option>
                  <option value="1">BB</option>
                </select>
              </label>
              <label class="field">
                <span>Stack</span>
                <input
                  value={stack()}
                  inputmode="decimal"
                  onInput={(event) => {
                    setStack(event.currentTarget.value);
                    clearSolveOutput();
                  }}
                />
              </label>
              <label class="field">
                <span>Small blind</span>
                <input
                  value={sb()}
                  inputmode="decimal"
                  onInput={(event) => {
                    setSb(event.currentTarget.value);
                    clearSolveOutput();
                  }}
                />
              </label>
              <label class="field">
                <span>Big blind</span>
                <input
                  value={bb()}
                  inputmode="decimal"
                  onInput={(event) => {
                    setBb(event.currentTarget.value);
                    clearSolveOutput();
                  }}
                />
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
                clearSolveOutput();
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
                    clearSolveOutput();
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
                presets={[200, 400, 600, 1000]}
                min={1}
                onChange={(value) => {
                  setIterations(value);
                  clearSolveOutput();
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
                  clearSolveOutput();
                }}
              />
            </div>
            <Show when={solveInputError() && !descriptorError()}>
              <p class="inline-error">{solveInputError()}</p>
            </Show>
            <button
              type="button"
              class="solve-button"
              disabled={isSolving() || !runtime() || Boolean(solveInputError())}
              onClick={() => void solve()}
            >
              <Play size={18} />
              <span>{isSolving() ? `Solving ${Math.floor(solveProgress() ?? 0)}%` : "Solve Spot"}</span>
            </button>
            <Show when={solveProgress() !== undefined}>
              <div class="progress-track solve-progress-track" aria-label="Solve progress">
                <div style={{ width: `${solveProgress() ?? 0}%` }} />
              </div>
            </Show>
          </div>
        </form>

        <section class="panel results">
          <div class="section-head">
            <h2>Results</h2>
            <Show when={solveResult()}>
              {(solved) => (
                <div class="metadata result-metadata">
                  <span>{solved().iterations} iterations</span>
                  <span>depth {solved().depth}</span>
                  <span>{solved().elapsedMs.toFixed(1)} ms</span>
                </div>
              )}
            </Show>
          </div>

          <Show when={solveError()}>
            <div class="notice error">
              <AlertTriangle size={18} />
              <span>{solveError()}</span>
            </div>
          </Show>

          <Show
            when={solveResult()}
            fallback={
              <Show
                when={isSolving()}
                fallback={<div class="empty-state">Run a solve to populate strategy and range output.</div>}
              >
                <div class="empty-state solving-state" aria-label="Solving">
                  <div class="spinner" />
                </div>
              </Show>
            }
          >
            {(solved) => (
              <>
                <Show
                  when={solved().result.actor === HERO_PLAYER}
                  fallback={
                    <div class="notice">
                      <AlertTriangle size={18} />
                      <span>
                        Villain is to act, so the selected hero hand is a blocker,
                        not an exact-hand policy row.
                      </span>
                    </div>
                  }
                >
                  <HeroPolicy
                    labels={solved().result.actionLabels}
                    legalMask={solved().result.legalMask}
                    policy={solved().result.policy}
                    handIndex={solved().heroHandIndex}
                    context={(descriptor() as StateDescriptor).finalContext}
                  />
                </Show>

                <RangeGridCollapses beliefs={solved().result.beliefsAtSpot} />
              </>
            )}
          </Show>
        </section>
      </section>
    </main>
  );
}

function HeroPolicy(props: {
  labels: readonly string[];
  legalMask: readonly number[];
  policy: Float32Array<ArrayBufferLike>;
  handIndex: number;
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
    <section class="subsection no-border hero-strategy">
      <h3>Hero strategy</h3>
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

function RangeGridCollapses(props: {
  beliefs: Float32Array<ArrayBufferLike>;
}): JSX.Element {
  const grids = createMemo(() => [
    buildRangeGrid(props.beliefs, HERO_PLAYER),
    buildRangeGrid(props.beliefs, 1),
  ]);
  return (
    <section class="subsection range-matrices">
      <For each={grids()}>{(grid) => <RangeGridCollapse grid={grid} />}</For>
    </section>
  );
}

function RangeGridCollapse(props: { grid: RangeGridModel }): JSX.Element {
  const [open, setOpen] = createSignal(false);
  return (
    <div class="range-collapse">
      <button
        type="button"
        class={`range-collapse-trigger ${open() ? "open" : ""}`}
        onClick={() => setOpen(!open())}
      >
        <span>{props.grid.label} range</span>
        <ChevronDown size={16} />
      </button>
      <Show when={open()}>
        <div class="range-matrix" role="grid" aria-label={`${props.grid.label} range grid`}>
          <For each={props.grid.rows}>
            {(row) => (
              <For each={row}>
                {(cell) => (
                  <button
                    type="button"
                    class={`range-matrix-cell ${cell.kind}`}
                    aria-label={cell.label}
                    title={cell.title}
                    style={`--range-alpha: ${cell.alpha};`}
                  />
                )}
              </For>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}

function buildRangeGrid(
  beliefs: Float32Array<ArrayBufferLike>,
  player: PlayerIndex,
): RangeGridModel {
  const offset = player * NUM_HANDS;
  const draftRows: Array<
    Array<RangeCell & { total: number; mean: number; top: RangeGridCombo[] }>
  > = [];
  let maxComboWeight = 0;

  for (const rowRank of RANKS) {
    const row: Array<RangeCell & { total: number; mean: number; top: RangeGridCombo[] }> = [];
    for (const colRank of RANKS) {
      const cell = rangeCell(rowRank, colRank);
      const exactCombos = rangeOptions(cell.key, UNBLOCKED_CARDS);
      const weightedCombos: RangeGridCombo[] = exactCombos.map((option) => {
        const c0 = parseCard(option.cards[0]);
        const c1 = parseCard(option.cards[1]);
        const weight = beliefs[offset + handComboIndex(c0, c1)] ?? 0;
        maxComboWeight = Math.max(maxComboWeight, weight);
        return { hand: `${option.cards[0]} ${option.cards[1]}`, weight };
      });
      const total = weightedCombos.reduce((sum, combo) => sum + combo.weight, 0);
      const mean = exactCombos.length > 0 ? total / exactCombos.length : 0;
      weightedCombos.sort((a, b) => b.weight - a.weight);
      row.push({ ...cell, total, mean, top: weightedCombos.slice(0, 3) });
    }
    draftRows.push(row);
  }

  const rows = draftRows.map((row) =>
    row.map((cell) => ({
      ...cell,
      alpha: maxComboWeight > 0 ? cell.mean / maxComboWeight : 0,
      title: rangeGridCellTitle(cell),
    })),
  );
  return { label: playerLabel(player), rows };
}

function rangeGridCellTitle(cell: RangeCell & {
  total: number;
  top: RangeGridCombo[];
}): string {
  const comboCount = rangeOptions(cell.key, UNBLOCKED_CARDS).length;
  const lines = [
    `${cell.label}: ${formatPercent(cell.total)} of range`,
    `Top 3 of ${comboCount} combos`,
  ];
  for (const combo of cell.top) {
    lines.push(`${combo.hand}: ${formatPercent(combo.weight)}`);
  }
  return lines.join("\n");
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default App;
