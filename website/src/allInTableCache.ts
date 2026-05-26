const DB_NAME = "p2-webgpu-cfr-allin-cache";
const STORE_NAME = "tables";
const DB_VERSION = 1;
const MAX_FLOP_TABLES = 5;

export type CachedAllInTableKind = "preflop" | "flop";

interface CachedAllInTableRecord {
  key: string;
  kind: CachedAllInTableKind;
  bytes: ArrayBuffer;
  updatedAt: number;
}

export async function readCachedAllInTable(key: string): Promise<ArrayBuffer | undefined> {
  const db = await openAllInCacheDb();
  if (!db) return undefined;
  try {
    const record = await getRecord(db, key);
    return record?.bytes;
  } finally {
    db.close();
  }
}

export async function writeCachedAllInTable(
  key: string,
  kind: CachedAllInTableKind,
  bytes: ArrayBuffer,
): Promise<void> {
  const db = await openAllInCacheDb();
  if (!db) return;
  try {
    await putRecord(db, {
      key,
      kind,
      bytes: bytes.slice(0),
      updatedAt: Date.now(),
    });
    if (kind === "flop") {
      await pruneFlopTables(db);
    }
  } finally {
    db.close();
  }
}

function openAllInCacheDb(): Promise<IDBDatabase | undefined> {
  if (!("indexedDB" in globalThis)) return Promise.resolve(undefined);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    request.onerror = () => reject(request.error ?? new Error("failed to open all-in table cache"));
    request.onsuccess = () => resolve(request.result);
  });
}

function getRecord(db: IDBDatabase, key: string): Promise<CachedAllInTableRecord | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).get(key);
    request.onerror = () => reject(request.error ?? new Error("failed to read all-in table cache"));
    request.onsuccess = () => resolve(request.result as CachedAllInTableRecord | undefined);
  });
}

function putRecord(db: IDBDatabase, record: CachedAllInTableRecord): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.onerror = () => reject(tx.error ?? new Error("failed to write all-in table cache"));
    tx.oncomplete = () => resolve();
    tx.objectStore(STORE_NAME).put(record);
  });
}

function getAllRecords(db: IDBDatabase): Promise<CachedAllInTableRecord[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onerror = () => reject(request.error ?? new Error("failed to list all-in table cache"));
    request.onsuccess = () => resolve(request.result as CachedAllInTableRecord[]);
  });
}

async function pruneFlopTables(db: IDBDatabase): Promise<void> {
  const records = (await getAllRecords(db))
    .filter((record) => record.kind === "flop")
    .sort((a, b) => b.updatedAt - a.updatedAt);
  const stale = records.slice(MAX_FLOP_TABLES);
  if (stale.length === 0) return;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.onerror = () => reject(tx.error ?? new Error("failed to prune all-in table cache"));
    tx.oncomplete = () => resolve();
    const store = tx.objectStore(STORE_NAME);
    for (const record of stale) {
      store.delete(record.key);
    }
  });
}
