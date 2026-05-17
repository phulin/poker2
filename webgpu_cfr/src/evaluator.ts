import {
  ACCUMULATE_REGRET_WGSL,
  ACTION_PROBS_WGSL,
  BELIEF_APPLY_WGSL,
  BELIEF_NORMALIZE_WGSL,
  FINALIZE_POLICY_WGSL,
  REGRET_MATCH_WGSL,
} from "./kernels.js";
import { readFloatBuffer } from "./gpuBuffers.js";
import type {
  EvaluationResult,
  LocalCfrProblem,
  LocalSolveResult,
  PlayerIndex,
  WebgpuCfrFixture,
} from "./types.js";

interface SolveReadOptions {
  readPolicy?: boolean;
  readActionProbs?: boolean;
}

interface TempBuffer {
  buffer: GPUBuffer;
  key: number;
  kind: "storage" | "uniform";
}

function u32Params(
  numHands: number,
  numActions: number,
  actor: PlayerIndex,
  selectedAction: number,
  iterations: number,
): Uint32Array<ArrayBuffer> {
  return new Uint32Array([
    numHands,
    numActions,
    actor,
    selectedAction,
    iterations,
    0,
    0,
    0,
  ]);
}

function asU32(values: number[]): Uint32Array {
  return Uint32Array.from(values.map((v) => (v ? 1 : 0)));
}

function assertProblem(problem: LocalCfrProblem, numHands: number, numActions: number) {
  if (problem.legalMask.length !== numActions) {
    throw new Error(
      `legalMask has ${problem.legalMask.length} entries, expected ${numActions}`,
    );
  }
  const expectedValues = numActions * 2 * numHands;
  if (problem.childValues.length !== expectedValues) {
    throw new Error(
      `childValues has ${problem.childValues.length} entries, expected ${expectedValues}`,
    );
  }
}

export class GpuCfrEvaluator {
  readonly device: GPUDevice;
  private readonly regretMatch: GPUComputePipeline;
  private readonly accumulateRegret: GPUComputePipeline;
  private readonly finalizePolicy: GPUComputePipeline;
  private readonly beliefApply: GPUComputePipeline;
  private readonly beliefNormalize: GPUComputePipeline;
  private readonly actionProbs: GPUComputePipeline;
  private readonly storagePool = new Map<number, GPUBuffer[]>();
  private readonly uniformPool = new Map<number, GPUBuffer[]>();
  private readonly zeroFloatArrays = new Map<number, Float32Array<ArrayBuffer>>();

  constructor(device: GPUDevice) {
    this.device = device;
    this.regretMatch = this.pipeline(REGRET_MATCH_WGSL, "regret-match");
    this.accumulateRegret = this.pipeline(ACCUMULATE_REGRET_WGSL, "accumulate-regret");
    this.finalizePolicy = this.pipeline(FINALIZE_POLICY_WGSL, "finalize-policy");
    this.beliefApply = this.pipeline(BELIEF_APPLY_WGSL, "belief-apply");
    this.beliefNormalize = this.pipeline(BELIEF_NORMALIZE_WGSL, "belief-normalize");
    this.actionProbs = this.pipeline(ACTION_PROBS_WGSL, "action-probs");
  }

  async evaluate(fixture: WebgpuCfrFixture): Promise<EvaluationResult> {
    const { numHands, numActions } = fixture;
    if (fixture.initialBeliefs.length !== 2 * numHands) {
      throw new Error(
        `initialBeliefs has ${fixture.initialBeliefs.length} entries, expected ${2 * numHands}`,
      );
    }
    if (fixture.problems.length === 0) {
      throw new Error("fixture must contain at least one CFR problem");
    }

    let beliefs: Float32Array<ArrayBufferLike> = new Float32Array(
      fixture.initialBeliefs,
    );
    let final: LocalSolveResult | undefined;

    for (const problem of fixture.problems) {
      assertProblem(problem, numHands, numActions);
      final = await this.solve(
        problem,
        beliefs,
        numHands,
        numActions,
        fixture.iterations,
        problem.action === undefined
          ? undefined
          : { readPolicy: false, readActionProbs: false },
      );
      if (problem.action !== undefined) {
        if (!final.beliefsAfter) {
          throw new Error("internal error: selected action did not produce beliefs");
        }
        beliefs = final.beliefsAfter;
      }
    }

    if (!final) {
      throw new Error("no final CFR result produced");
    }
    return {
      beliefsAtSpot: beliefs,
      actionProbs: final.actionProbs,
      policy: final.policy,
    };
  }

  async solve(
    problem: LocalCfrProblem,
    beliefs: Float32Array<ArrayBufferLike>,
    numHands: number,
    numActions: number,
    iterations: number,
    readOptions: SolveReadOptions = {},
  ): Promise<LocalSolveResult> {
    assertProblem(problem, numHands, numActions);
    const childValues = this.acquireStorageFromData(
      new Float32Array(problem.childValues),
    );
    return await this.solvePrepared(
      problem,
      beliefs,
      numHands,
      numActions,
      iterations,
      childValues.buffer,
      childValues,
      readOptions,
    );
  }

  async solveGpuChildValues(
    problem: Omit<LocalCfrProblem, "childValues">,
    beliefs: Float32Array<ArrayBufferLike>,
    numHands: number,
    numActions: number,
    iterations: number,
    childValues: GPUBuffer,
    readOptions: SolveReadOptions = {},
  ): Promise<LocalSolveResult> {
    if (problem.legalMask.length !== numActions) {
      throw new Error(
        `legalMask has ${problem.legalMask.length} entries, expected ${numActions}`,
      );
    }
    return await this.solvePrepared(
      problem,
      beliefs,
      numHands,
      numActions,
      iterations,
      childValues,
      undefined,
      readOptions,
    );
  }

  private async solvePrepared(
    problem: Omit<LocalCfrProblem, "childValues">,
    beliefs: Float32Array<ArrayBufferLike>,
    numHands: number,
    numActions: number,
    iterations: number,
    childValues: GPUBuffer,
    ownedChildValues: TempBuffer | undefined,
    readOptions: SolveReadOptions = {},
  ): Promise<LocalSolveResult> {
    const readPolicy = readOptions.readPolicy ?? true;
    const readActionProbs = readOptions.readActionProbs ?? true;
    const debug =
      typeof process !== "undefined" && process.env?.WEBGPU_CFR_DEBUG === "1";
    if (debug) console.error("solve:start");
    const totalPolicy = numHands * numActions;
    const workgroupsHands = Math.ceil(numHands / 64);
    const workgroupsPolicy = Math.ceil(totalPolicy / 64);

    const temps: TempBuffer[] = [];
    if (ownedChildValues) {
      temps.push(ownedChildValues);
    }
    const storage = (
      data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
    ): GPUBuffer => {
      const temp = this.acquireStorageFromData(data);
      temps.push(temp);
      return temp.buffer;
    };
    const zeroed = (elements: number): GPUBuffer => {
      const temp = this.acquireZeroedStorage(elements);
      temps.push(temp);
      return temp.buffer;
    };
    const uniform = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): GPUBuffer => {
      const temp = this.acquireUniform(data);
      temps.push(temp);
      return temp.buffer;
    };

    try {
      const legal = storage(asU32(problem.legalMask));
      const regrets = zeroed(totalPolicy);
      const policy = zeroed(totalPolicy);
      const avgPolicy = zeroed(totalPolicy);
      const params = uniform(
        u32Params(
          numHands,
          numActions,
          problem.actor,
          problem.action ?? 0,
          iterations,
        ),
      );
      const bindGroup = (
        pipeline: GPUComputePipeline,
        entries: GPUBindGroupEntry[],
      ) =>
        this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });

      const regretMatchBindGroup = bindGroup(this.regretMatch, [
        { binding: 0, resource: { buffer: legal } },
        { binding: 2, resource: { buffer: regrets } },
        { binding: 3, resource: { buffer: policy } },
        { binding: 5, resource: { buffer: params } },
      ]);
      const accumulateRegretBindGroup = bindGroup(this.accumulateRegret, [
        { binding: 0, resource: { buffer: legal } },
        { binding: 1, resource: { buffer: childValues } },
        { binding: 2, resource: { buffer: regrets } },
        { binding: 3, resource: { buffer: policy } },
        { binding: 4, resource: { buffer: avgPolicy } },
        { binding: 5, resource: { buffer: params } },
      ]);
      const finalizePolicyBindGroup = bindGroup(this.finalizePolicy, [
        { binding: 3, resource: { buffer: policy } },
        { binding: 4, resource: { buffer: avgPolicy } },
        { binding: 5, resource: { buffer: params } },
      ]);

      const beliefsIn = storage(beliefs);
      let actionProbs: GPUBuffer | undefined;
      let actionBindGroup: GPUBindGroup | undefined;
      if (readActionProbs) {
        actionProbs = zeroed(numActions);
        actionBindGroup = this.device.createBindGroup({
          layout: this.actionProbs.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: beliefsIn } },
            { binding: 1, resource: { buffer: policy } },
            { binding: 2, resource: { buffer: actionProbs } },
            { binding: 3, resource: { buffer: params } },
          ],
        });
      }

      let beliefsOut: GPUBuffer | undefined;
      let applyBindGroup: GPUBindGroup | undefined;
      let normalizeBindGroup: GPUBindGroup | undefined;
      if (problem.action !== undefined) {
        beliefsOut = zeroed(2 * numHands);
        const denom = zeroed(1);
        applyBindGroup = this.device.createBindGroup({
          layout: this.beliefApply.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: beliefsIn } },
            { binding: 1, resource: { buffer: policy } },
            { binding: 2, resource: { buffer: beliefsOut } },
            { binding: 3, resource: { buffer: denom } },
            { binding: 4, resource: { buffer: params } },
          ],
        });
        normalizeBindGroup = this.device.createBindGroup({
          layout: this.beliefNormalize.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: beliefsIn } },
            { binding: 2, resource: { buffer: beliefsOut } },
            { binding: 3, resource: { buffer: denom } },
            { binding: 4, resource: { buffer: params } },
          ],
        });
      }

      if (debug) console.error("solve:encode");
      const encoder = this.device.createCommandEncoder();
      for (let i = 0; i < iterations; i += 1) {
        this.encodeCompute(
          encoder,
          this.regretMatch,
          regretMatchBindGroup,
          workgroupsHands,
          debug ? "regret-match" : undefined,
        );
        this.encodeCompute(
          encoder,
          this.accumulateRegret,
          accumulateRegretBindGroup,
          workgroupsHands,
          debug ? "accumulate-regret" : undefined,
        );
      }
      this.encodeCompute(
        encoder,
        this.finalizePolicy,
        finalizePolicyBindGroup,
        workgroupsPolicy,
        debug ? "finalize-policy" : undefined,
      );
      if (actionBindGroup) {
        this.encodeCompute(
          encoder,
          this.actionProbs,
          actionBindGroup,
          numActions,
          debug ? "action-probs" : undefined,
        );
      }

      if (applyBindGroup && normalizeBindGroup) {
        this.encodeCompute(
          encoder,
          this.beliefApply,
          applyBindGroup,
          1,
          debug ? "belief-apply" : undefined,
        );
        this.encodeCompute(
          encoder,
          this.beliefNormalize,
          normalizeBindGroup,
          workgroupsHands,
          debug ? "belief-normalize" : undefined,
        );
      }
      if (debug) console.error("solve:submit");
      this.device.queue.submit([encoder.finish()]);
      if (debug) console.error("solve:read-policy");
      const policyData = readPolicy
        ? await readFloatBuffer(this.device, policy, totalPolicy)
        : new Float32Array(0);
      if (debug) console.error("solve:read-actions");
      const actionProbData =
        readActionProbs && actionProbs
          ? await readFloatBuffer(this.device, actionProbs, numActions)
          : new Float32Array(0);
      if (debug) console.error("solve:read-beliefs");
      const beliefsAfter = beliefsOut
        ? await readFloatBuffer(this.device, beliefsOut, 2 * numHands)
        : undefined;

      const result: LocalSolveResult = {
        policy: policyData,
        actionProbs: actionProbData,
      };
      if (beliefsAfter !== undefined) {
        result.beliefsAfter = beliefsAfter;
      }
      if (debug) console.error("solve:done");
      return result;
    } finally {
      if (debug) console.error("solve:release");
      for (const temp of temps) {
        this.releaseTemp(temp);
      }
    }
  }

  private pipeline(source: string, label: string): GPUComputePipeline {
    return this.device.createComputePipeline({
      label,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ label: `${label}.wgsl`, code: source }),
        entryPoint: "main",
      },
    });
  }

  private async runCompute(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    x: number,
    label?: string,
  ): Promise<void> {
    if (label) console.error(`solve:run:${label}`);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x);
    pass.end();
    if (label) console.error(`solve:submit:${label}`);
    this.device.queue.submit([encoder.finish()]);
    if (label) console.error(`solve:wait:${label}`);
    await this.device.queue.onSubmittedWorkDone();
    if (label) console.error(`solve:done:${label}`);
  }

  private encodeCompute(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    x: number,
    label?: string,
  ): void {
    if (label) console.error(`solve:encode:${label}`);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x);
    pass.end();
  }

  private acquireStorageFromData(
    data: Float32Array<ArrayBufferLike> | Uint32Array<ArrayBufferLike>,
  ): TempBuffer {
    const temp = this.acquireStorageBytes(data.byteLength);
    this.device.queue.writeBuffer(
      temp.buffer,
      0,
      data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
    );
    return temp;
  }

  private acquireZeroedStorage(elements: number): TempBuffer {
    const temp = this.acquireStorageBytes(elements * Float32Array.BYTES_PER_ELEMENT);
    this.device.queue.writeBuffer(temp.buffer, 0, this.zeroFloatArray(elements));
    return temp;
  }

  private acquireStorageBytes(byteLength: number): TempBuffer {
    const key = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const buffer = this.storagePool.get(key)?.pop();
    return {
      buffer:
        buffer ??
        this.device.createBuffer({
          size: key,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        }),
      key,
      kind: "storage",
    };
  }

  private acquireUniform(
    data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
  ): TempBuffer {
    const key = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const buffer = this.uniformPool.get(key)?.pop();
    const temp: TempBuffer = {
      buffer:
        buffer ??
        this.device.createBuffer({
          size: key,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      key,
      kind: "uniform",
    };
    this.device.queue.writeBuffer(temp.buffer, 0, data);
    return temp;
  }

  private zeroFloatArray(elements: number): Float32Array<ArrayBuffer> {
    let zeroes = this.zeroFloatArrays.get(elements);
    if (!zeroes) {
      zeroes = new Float32Array(elements);
      this.zeroFloatArrays.set(elements, zeroes);
    }
    return zeroes;
  }

  private releaseTemp(temp: TempBuffer): void {
    const pool = temp.kind === "storage" ? this.storagePool : this.uniformPool;
    const buffers = pool.get(temp.key);
    if (buffers) {
      buffers.push(temp.buffer);
    } else {
      pool.set(temp.key, [temp.buffer]);
    }
  }

  dispose(): void {
    for (const pool of [this.storagePool, this.uniformPool]) {
      for (const buffers of pool.values()) {
        for (const buffer of buffers) {
          buffer.destroy();
        }
      }
      pool.clear();
    }
    this.zeroFloatArrays.clear();
  }
}

export async function evaluateFixture(
  device: GPUDevice,
  fixture: WebgpuCfrFixture,
): Promise<EvaluationResult> {
  const evaluator = new GpuCfrEvaluator(device);
  try {
    return await evaluator.evaluate(fixture);
  } finally {
    evaluator.dispose();
  }
}
