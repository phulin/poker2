import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { WebGpuAllInTableGenerator } from "../src/allInTableGenerator.js";
import { ALL_IN_I16_SCALE, createManifestAllInTableProvider } from "../src/allInTables.js";
import { createDawnDevice } from "../src/gpu.js";
import { DEFAULT_FORCE_DECK, PublicHunlEnv } from "../src/hunlEnv.js";

function flopEnv(board: readonly number[]): PublicHunlEnv {
  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins: [],
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  env.street = 1;
  env.boardIndices = [board[0]!, board[1]!, board[2]!, -1, -1];
  return env;
}

test("manifest all-in provider skips streets without table metadata and no device", async () => {
  const provider = createManifestAllInTableProvider(
    {
      allIn: {
        enabled: true,
        preflop: {
          file: "/allin/preflop.i16",
          dtype: "int16",
        },
      },
    },
    "https://example.test/model.json",
    async () => {
      throw new Error("missing street metadata should not load assets");
    },
  );
  assert.ok(provider);

  assert.equal(await provider.tableForRoot(flopEnv([0, 13, 26])), undefined);
});

test("manifest all-in provider resolves local root-relative all-in paths from public root", async () => {
  const loadedUrls: string[] = [];
  const provider = createManifestAllInTableProvider(
    {
      allIn: {
        enabled: true,
        preflop: {
          file: "/allin/preflop.i16",
          dtype: "int16",
        },
      },
    },
    "file:///tmp/p2/website/public/models/rebel_latest/model.json",
    async (url) => {
      loadedUrls.push(url);
      return new Int16Array([0, 1]).buffer;
    },
  );
  assert.ok(provider);

  const env = new PublicHunlEnv({
    stack: 20,
    sb: 1,
    bb: 2,
    betBins: [],
    button: 1,
    forceDeck: DEFAULT_FORCE_DECK,
  });
  assert.equal((await provider.tableForRoot(env))?.street, 0);
  assert.deepEqual(loadedUrls, ["file:///tmp/p2/website/public/allin/preflop.i16"]);
});

test("manifest all-in provider does not generate missing flop metadata while online", async (t) => {
  let device: GPUDevice;
  try {
    device = await createDawnDevice();
  } catch (error) {
    t.skip(`WebGPU device unavailable: ${error}`);
    return;
  }
  try {
    const provider = createManifestAllInTableProvider(
      {
        allIn: {
          enabled: true,
          preflop: {
            file: "/allin/preflop.i16",
            dtype: "int16",
          },
        },
      },
      "https://example.test/model.json",
      async () => {
        throw new Error("missing flop metadata should not load assets");
      },
      device,
      { isOffline: () => false },
    );
    assert.ok(provider);

    assert.equal(await provider.tableForRoot(flopEnv([0, 13, 26])), undefined);
  } finally {
    device.destroy();
  }
});

function pythonGeneratedTable(board: readonly number[]): Int16Array<ArrayBuffer> {
  const result = spawnSync(
    "uv",
    [
      "run",
      "python",
      "-c",
      [
        "import sys, torch",
        "from p2.search.allin_payoff import compute_postflop_payoff_quantized",
        "board = torch.tensor([int(x) for x in sys.argv[1].split(',')], dtype=torch.long)",
        "table = compute_postflop_payoff_quantized(board, dtype=torch.int16).cpu().numpy()",
        "sys.stdout.buffer.write(table.astype('<i2', copy=False).tobytes(order='C'))",
      ].join("; "),
      board.join(","),
    ],
    {
      cwd: new URL("../..", import.meta.url),
      maxBuffer: 16 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(`python all-in table generation failed: ${result.stderr.toString()}`);
  }
  const bytes = new Uint8Array(
    result.stdout.buffer,
    result.stdout.byteOffset,
    result.stdout.byteLength,
  ).slice();
  return new Int16Array(bytes.buffer);
}

function assertSameTable(
  actual: Int16Array<ArrayBufferLike>,
  expected: Int16Array<ArrayBufferLike>,
): void {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    if (actual[i] !== expected[i]) {
      assert.fail(`table mismatch at ${i}: actual=${actual[i]} expected=${expected[i]}`);
    }
  }
}

test("WebGPU all-in table generator matches Python turn table", async (t) => {
  let device: GPUDevice;
  try {
    device = await createDawnDevice();
  } catch (error) {
    t.skip(`WebGPU device unavailable: ${error}`);
    return;
  }
  try {
    const board = [0, 13, 26, 39];
    const generator = new WebGpuAllInTableGenerator(device);
    assertSameTable(await generator.tableForBoard(board), pythonGeneratedTable(board));
  } finally {
    device.destroy();
  }
});

test("manifest all-in provider generates missing flop table metadata only while offline", async (t) => {
  let device: GPUDevice;
  try {
    device = await createDawnDevice();
  } catch (error) {
    t.skip(`WebGPU device unavailable: ${error}`);
    return;
  }
  try {
    const board = [0, 13, 26];
    const provider = createManifestAllInTableProvider(
      {
        allIn: {
          enabled: true,
          preflop: {
            file: "/allin/preflop.i16",
            dtype: "int16",
          },
        },
      },
      "https://example.test/model.json",
      async () => {
        throw new Error("missing flop metadata should generate without loading assets");
      },
      device,
      { isOffline: () => true },
    );
    assert.ok(provider);

    const context = await provider.tableForRoot(flopEnv(board));
    assert.ok(context);
    assert.equal(context.street, 1);
    assert.equal(context.scale, ALL_IN_I16_SCALE);
    assertSameTable(context.table, pythonGeneratedTable(board));
  } finally {
    device.destroy();
  }
});
