import assert from 'node:assert/strict';
import {
  appendHistoryEntry,
  inferHistoryLabel,
  normalizeEditHistory,
  serializeEditHistory,
} from './editHistory';
import { Adjustments, INITIAL_ADJUSTMENTS } from './adjustments';

const withAdjustments = (patch: Partial<Adjustments>): Adjustments => ({
  ...structuredClone(INITIAL_ADJUSTMENTS),
  ...patch,
});

export function runEditHistoryTests() {
  const original = withAdjustments({});
  const exposureEdit = withAdjustments({ exposure: 1.25 });

  const singleAdjustment = normalizeEditHistory(undefined, exposureEdit, original);
  assert.equal(singleAdjustment.entries.length, 2);
  assert.equal(singleAdjustment.currentIndex, 1);
  assert.equal(singleAdjustment.entries[1].label, 'Current Edit');
  assert.equal(singleAdjustment.entries[1].adjustments.exposure, 1.25);

  const manualLabel = inferHistoryLabel(original, exposureEdit);
  assert.equal(manualLabel, 'Exposure');

  const contrastEdit = withAdjustments({ exposure: 1.25, contrast: 14 });
  const multipleManual = appendHistoryEntry(
    singleAdjustment.entries,
    singleAdjustment.currentIndex,
    contrastEdit,
    inferHistoryLabel(exposureEdit, contrastEdit),
  );
  assert.equal(multipleManual.entries[multipleManual.currentIndex].label, 'Contrast');
  assert.equal(multipleManual.entries[multipleManual.currentIndex].adjustments.contrast, 14);

  const presetEdit = withAdjustments({ exposure: 0.5, contrast: 10, vibrance: 20 });
  const presetHistory = appendHistoryEntry(
    multipleManual.entries,
    multipleManual.currentIndex,
    presetEdit,
    'Preset: Clean Color',
  );
  assert.equal(presetHistory.entries[presetHistory.currentIndex].label, 'Preset: Clean Color');

  const autoEdit = withAdjustments({ exposure: 0.25, highlights: -20, shadows: 15 });
  const autoHistory = appendHistoryEntry(presetHistory.entries, presetHistory.currentIndex, autoEdit, 'Auto Adjustment');
  assert.equal(autoHistory.entries[autoHistory.currentIndex].label, 'Auto Adjustment');
  assert.equal(autoHistory.entries[autoHistory.currentIndex].adjustments.highlights, -20);

  const aiEdit = withAdjustments({
    aiPatches: [
      {
        id: 'patch-1',
        invert: false,
        isLoading: false,
        name: 'Remove sign',
        patchData: { assetId: 'generated-asset' },
        prompt: 'clean wall',
        subMasks: [],
        visible: true,
      },
    ],
  });
  const aiHistory = appendHistoryEntry(autoHistory.entries, autoHistory.currentIndex, aiEdit, 'Generative Replace');
  assert.equal(aiHistory.entries[aiHistory.currentIndex].label, 'Generative Replace');
  assert.equal(aiHistory.entries[aiHistory.currentIndex].adjustments.aiPatches[0].patchData.assetId, 'generated-asset');

  const branchEdit = withAdjustments({ exposure: -0.75 });
  const branched = appendHistoryEntry(aiHistory.entries, 2, branchEdit, 'Exposure');
  assert.equal(branched.entries.length, 4);
  assert.equal(branched.currentIndex, 3);
  assert.equal(branched.entries[3].adjustments.exposure, -0.75);

  const serialized = serializeEditHistory(aiHistory.entries, aiHistory.currentIndex);
  const restored = normalizeEditHistory(serialized, aiEdit, original);
  assert.equal(restored.entries.length, aiHistory.entries.length);
  assert.equal(restored.currentIndex, aiHistory.currentIndex);
  assert.deepEqual(restored.entries[restored.currentIndex].adjustments.aiPatches, aiEdit.aiPatches);

  const olderSidecar = normalizeEditHistory(undefined, original, original);
  assert.equal(olderSidecar.entries.length, 1);
  assert.equal(olderSidecar.entries[0].label, 'Original');

  const malformed = normalizeEditHistory({ currentIndex: 99, entries: [{ label: '', adjustments: null }] }, exposureEdit, original);
  assert.equal(malformed.currentIndex, 0);
  assert.equal(malformed.entries[0].label, 'Edit');
  assert.equal(malformed.entries[0].adjustments.exposure, exposureEdit.exposure);
}
