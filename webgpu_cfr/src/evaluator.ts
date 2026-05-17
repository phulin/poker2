import {
  ACCUMULATE_REGRET_WGSL,
  ACTION_PROBS_WGSL,
  BELIEF_APPLY_WGSL,
  BELIEF_NORMALIZE_WGSL,
  FINALIZE_POLICY_WGSL,
  REGRET_MATCH_WGSL,
} from "./kernels.js";
import {
  makeStorageBuffer,
  makeUniformBuffer,
  readFloatBuffer,
} from "./gpuBuffers.js";
import type {
  EvaluationResult,
  LocalCfrProblem,
  LocalSolveResult,
  PlayerIndex,
  WebgpuCfrFixture,
} from "./types.js";

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
      final = await this.solve(problem, beliefs, numHands, numActions, fixture.iterations);
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
  ): Promise<LocalSolveResult> {
    const debug =
      typeof process !== "undefined" && process.env?.WEBGPU_CFR_DEBUG === "1";
    if (debug) console.error("solve:start");
    const totalPolicy = numHands * numActions;
    const workgroupsHands = Math.ceil(numHands / 64);
    const workgroupsPolicy = Math.ceil(totalPolicy / 64);

    const legal = makeStorageBuffer(this.device, asU32(problem.legalMask));
    const childValues = makeStorageBuffer(
      this.device,
      new Float32Array(problem.childValues),
    );
    const regrets = makeStorageBuffer(this.device, new Float32Array(totalPolicy));
    const policy = makeStorageBuffer(this.device, new Float32Array(totalPolicy));
    const avgPolicy = makeStorageBuffer(this.device, new Float32Array(totalPolicy));
    const params = makeUniformBuffer(
      this.device,
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

    const beliefsIn = makeStorageBuffer(this.device, beliefs);
    const actionProbs = makeStorageBuffer(this.device, new Float32Array(numActions));
    const actionBindGroup = this.device.createBindGroup({
      layout: this.actionProbs.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: beliefsIn } },
        { binding: 1, resource: { buffer: policy } },
        { binding: 2, resource: { buffer: actionProbs } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    let beliefsOut: GPUBuffer | undefined;
    let denom: GPUBuffer | undefined;
    let applyBindGroup: GPUBindGroup | undefined;
    let normalizeBindGroup: GPUBindGroup | undefined;
    if (problem.action !== undefined) {
      beliefsOut = makeStorageBuffer(this.device, new Float32Array(2 * numHands));
      denom = makeStorageBuffer(this.device, new Float32Array([0]));
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
    this.encodeCompute(
      encoder,
      this.actionProbs,
      actionBindGroup,
      numActions,
      debug ? "action-probs" : undefined,
    );

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

    const policyData = await readFloatBuffer(this.device, policy, totalPolicy);
    if (debug) console.error("solve:read-actions");
    const actionProbData = await readFloatBuffer(this.device, actionProbs, numActions);
    if (debug) console.error("solve:read-beliefs");
    const beliefsAfter = beliefsOut
      ? await readFloatBuffer(this.device, beliefsOut, 2 * numHands)
      : undefined;
    if (debug) console.error("solve:destroy");

    for (const buffer of [
      legal,
      childValues,
      regrets,
      policy,
      avgPolicy,
      params,
      beliefsIn,
      actionProbs,
      beliefsOut,
      denom,
    ]) {
      buffer?.destroy();
    }

    const result: LocalSolveResult = {
      policy: policyData,
      actionProbs: actionProbData,
    };
    if (beliefsAfter !== undefined) {
      result.beliefsAfter = beliefsAfter;
    }
    if (debug) console.error("solve:done");
    return result;
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
}

export async function evaluateFixture(
  device: GPUDevice,
  fixture: WebgpuCfrFixture,
): Promise<EvaluationResult> {
  return await new GpuCfrEvaluator(device).evaluate(fixture);
}
