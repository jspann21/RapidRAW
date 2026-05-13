import { build } from 'esbuild';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const tempDir = await mkdtemp(path.join(tmpdir(), 'rapidraw-history-tests-'));
const outfile = path.join(tempDir, 'editHistory.test.mjs');

try {
  await build({
    bundle: true,
    entryPoints: ['src/utils/editHistory.test.ts'],
    format: 'esm',
    outfile,
    platform: 'node',
  });

  const testModule = await import(pathToFileURL(outfile).href);
  await testModule.runEditHistoryTests();
  console.log('edit history tests passed');
} finally {
  await rm(tempDir, { force: true, recursive: true });
}
