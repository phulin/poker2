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
    ffnDim: number;
    numHiddenLayers: number;
    numPolicyLayers: number;
    numValueLayers: number;
    sharedTrunk: boolean;
    enforceZeroSum: boolean;
    nonlinearity: "swiglu";
    normalization: "rmsnorm";
  };
  env: {
    stack: number;
    sb: number;
    bb: number;
    betBins: number[];
    flopShowdown: boolean;
    defaultButton: PlayerIndex;
    defaultForceDeck: number[];
  };
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
  iterations: number;
  initialState?: BrowserCfrInitialState;
}

export interface BrowserEvaluationResult extends EvaluationResult {
  actionLabels: string[];
}
