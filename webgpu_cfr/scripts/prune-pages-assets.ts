#!/usr/bin/env tsx
import { rm } from "node:fs/promises";
import { resolve } from "node:path";

const appModelsDir = resolve("dist/app/models");

await rm(appModelsDir, { force: true, recursive: true });
