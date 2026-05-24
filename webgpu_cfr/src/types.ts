export type PlayerIndex = 0 | 1;

export interface LocalCfrProblem {
  actor: PlayerIndex;
  action?: number;
  legalMask: number[];
  childValues: ArrayLike<number>;
}

export interface WebgpuCfrFixture {
  schemaVersion: 1;
  source: string;
  snapshot: string;
  spot: number[];
  iterations: number;
  numHands: number;
  numActions: number;
  actionLabels: string[];
  initialBeliefs: number[];
  problems: LocalCfrProblem[];
  expected?: {
    beliefsAtSpot: number[];
    actionProbs: number[];
    policy: number[];
  };
}

export interface LocalSolveResult {
  policy: Float32Array<ArrayBufferLike>;
  actionProbs: Float32Array<ArrayBufferLike>;
  beliefsAfter?: Float32Array<ArrayBufferLike>;
  beliefsAfterBuffer?: GPUBuffer;
  releaseBeliefsAfterBuffer?: () => void;
}

export interface EvaluationResult {
  beliefsAtSpot: Float32Array<ArrayBufferLike>;
  actionProbs: Float32Array<ArrayBufferLike>;
  policy: Float32Array<ArrayBufferLike>;
}

export interface BetterFfnTensorManifest {
  name: string;
  shape: number[];
  dtype: "float32";
  byteOffset: number;
  byteLength: number;
  sha256: string;
}

export interface BetterFfnCfrManifest {
  source?: string;
  enabled?: boolean;
  depth?: number;
  iterations?: number;
  iterationsStart?: number;
  iterationsFinal?: number | null;
  scheduleProgress?: number | null;
  warmStartIterations?: number;
  warmStartType?: string;
  warmStartMultiplier?: number;
  branching?: number;
  beliefSamples?: number;
  sampleEpsilon?: number;
  dcfrAlpha?: number;
  dcfrAlphaFinal?: number | null;
  dcfrBeta?: number;
  dcfrBetaFinal?: number | null;
  dcfrGamma?: number;
  dcfrGammaFinal?: number | null;
  dcfrPlusDelay?: number;
  dcfrPlusDelayConfigured?: number;
  includeAveragePolicy?: boolean;
  cfrType?: string;
  cfrPlus?: boolean;
  cfrAvg?: boolean;
  sparse?: boolean;
  sparseFused?: boolean;
  valueTargetsFromFinalPolicy?: boolean;
}

export interface BetterFfnManifest {
  schemaVersion: 1;
  format: "p2.better_ffn.webgpu";
  source: {
    snapshot: string;
    exporter: string;
  };
  architecture: {
    numHands: number;
    numPlayers: 2;
    numActions: number;
    hiddenDim: number;
    rangeHiddenDim: number;
    boardInteractionDim: number;
    ffnDim: number;
    numHiddenLayers: number;
    numPolicyLayers: number;
    numValueLayers: number;
    sharedTrunk: boolean;
    enforceZeroSum: boolean;
    nonlinearity: "leaky_relu";
    normalization: "rmsnorm";
    contextDim?: number;
    policyRank?: number;
    policyHandBiasRank?: number;
    splitPolicyValue?: boolean;
  };
  env: {
    stack: number;
    sb: number;
    bb: number;
    betBins: number[];
    flopShowdown: boolean;
    maxStackBb?: number;
    defaultButton: PlayerIndex;
    defaultForceDeck: number[];
  };
  cfr?: BetterFfnCfrManifest;
  actionLabels: string[];
  tensors: BetterFfnTensorManifest[];
  weights: {
    file: string;
    byteLength: number;
    sha256: string;
  };
}

export interface BrowserCfrInitialState {
  button?: PlayerIndex;
  stack?: number;
  sb?: number;
  bb?: number;
  betBins?: number[];
  forceDeck?: number[];
  flopShowdown?: boolean;
}

export interface EvaluateSpotRequest {
  spot: number[];
  iterations?: number;
  depth?: number;
  cfrAvg?: boolean;
  initialState?: BrowserCfrInitialState;
  initialBeliefs?: Float32Array<ArrayBufferLike> | ArrayLike<number>;
  publicCards?: number[];
  heroPlayer?: PlayerIndex;
  heroHand?: [number, number];
  readPolicy?: boolean;
  readActionProbs?: boolean;
  readBeliefs?: boolean;
}

export interface BrowserEvaluationResult extends EvaluationResult {
  actionLabels: string[];
  legalMask: number[];
  actor: PlayerIndex;
}
