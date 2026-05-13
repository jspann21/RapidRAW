import { v4 as uuidv4 } from 'uuid';
import { Adjustments, INITIAL_ADJUSTMENTS, normalizeLoadedAdjustments } from './adjustments';

export const EDIT_HISTORY_VERSION = 1;
export const EDIT_HISTORY_LIMIT = 100;

export interface EditHistoryEntry {
  id: string;
  label: string;
  timestamp: string;
  adjustments: Adjustments;
}

export interface EditHistoryState {
  version: number;
  currentIndex: number;
  entries: EditHistoryEntry[];
}

const GEOMETRY_KEYS = new Set([
  'crop',
  'aspectRatio',
  'rotation',
  'flipHorizontal',
  'flipVertical',
  'orientationSteps',
  'transformDistortion',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
]);

const LABEL_BY_KEY: Record<string, string> = {
  aiPatches: 'AI Edit',
  masks: 'Masking',
  exposure: 'Exposure',
  brightness: 'Brightness',
  contrast: 'Contrast',
  highlights: 'Highlights',
  shadows: 'Shadows',
  whites: 'Whites',
  blacks: 'Blacks',
  temperature: 'Temperature',
  tint: 'Tint',
  saturation: 'Saturation',
  vibrance: 'Vibrance',
  colorGrading: 'Color Grading',
  colorCalibration: 'Color Calibration',
  hsl: 'Color Mixer',
  curves: 'Curves',
  pointCurves: 'Curves',
  parametricCurve: 'Curves',
  curveMode: 'Curves',
  clarity: 'Clarity',
  structure: 'Structure',
  dehaze: 'Dehaze',
  sharpness: 'Sharpness',
  sharpnessThreshold: 'Sharpness',
  lumaNoiseReduction: 'Noise Reduction',
  colorNoiseReduction: 'Noise Reduction',
  chromaticAberrationRedCyan: 'Chromatic Aberration',
  chromaticAberrationBlueYellow: 'Chromatic Aberration',
  vignetteAmount: 'Vignette',
  vignetteFeather: 'Vignette',
  vignetteMidpoint: 'Vignette',
  vignetteRoundness: 'Vignette',
  grainAmount: 'Grain',
  grainRoughness: 'Grain',
  grainSize: 'Grain',
  glowAmount: 'Glow',
  halationAmount: 'Halation',
  flareAmount: 'Flare',
  lutData: 'LUT',
  lutIntensity: 'LUT',
  lutName: 'LUT',
  lutPath: 'LUT',
  lutSize: 'LUT',
  lensMaker: 'Lens Correction',
  lensModel: 'Lens Correction',
  lensDistortionAmount: 'Lens Correction',
  lensVignetteAmount: 'Lens Correction',
  lensTcaAmount: 'Lens Correction',
  lensDistortionEnabled: 'Lens Correction',
  lensTcaEnabled: 'Lens Correction',
  lensVignetteEnabled: 'Lens Correction',
  lensDistortionParams: 'Lens Correction',
  toneMapper: 'Tone Mapper',
};

export const cloneAdjustments = (adjustments: Adjustments): Adjustments => structuredClone(adjustments);

export const areAdjustmentsEqual = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);

export const createHistoryEntry = (
  label: string,
  adjustments: Adjustments,
  timestamp: string = new Date().toISOString(),
): EditHistoryEntry => ({
  id: uuidv4(),
  label,
  timestamp,
  adjustments: cloneAdjustments(adjustments),
});

const safeNormalizeAdjustments = (value: unknown, fallback: Adjustments): Adjustments => {
  if (!value || typeof value !== 'object') {
    return cloneAdjustments(fallback);
  }

  try {
    return normalizeLoadedAdjustments(value as Adjustments);
  } catch (error) {
    console.warn('Ignoring malformed edit history adjustment snapshot:', error);
    return cloneAdjustments(fallback);
  }
};

export const normalizeEditHistory = (
  rawHistory: unknown,
  currentAdjustments: Adjustments,
  initialAdjustments: Adjustments = INITIAL_ADJUSTMENTS,
): EditHistoryState => {
  if (rawHistory && typeof rawHistory === 'object') {
    const history = rawHistory as Partial<EditHistoryState>;
    const rawEntries = Array.isArray(history.entries) ? history.entries : [];
    const entries = rawEntries
      .filter((entry): entry is Partial<EditHistoryEntry> => !!entry && typeof entry === 'object')
      .map((entry) =>
        createHistoryEntry(
          typeof entry.label === 'string' && entry.label.trim() ? entry.label : 'Edit',
          safeNormalizeAdjustments(entry.adjustments, currentAdjustments),
          typeof entry.timestamp === 'string' ? entry.timestamp : new Date().toISOString(),
        ),
      );

    if (entries.length > 0) {
      const requestedIndex = Number.isInteger(history.currentIndex) ? Number(history.currentIndex) : entries.length - 1;
      const currentIndex = Math.max(0, Math.min(entries.length - 1, requestedIndex));
      return {
        version: typeof history.version === 'number' ? history.version : EDIT_HISTORY_VERSION,
        currentIndex,
        entries,
      };
    }
  }

  const base = createHistoryEntry('Original', initialAdjustments);
  if (areAdjustmentsEqual(currentAdjustments, initialAdjustments)) {
    return { version: EDIT_HISTORY_VERSION, currentIndex: 0, entries: [base] };
  }

  return {
    version: EDIT_HISTORY_VERSION,
    currentIndex: 1,
    entries: [base, createHistoryEntry('Current Edit', currentAdjustments)],
  };
};

export const serializeEditHistory = (entries: EditHistoryEntry[], currentIndex: number): EditHistoryState => ({
  version: EDIT_HISTORY_VERSION,
  currentIndex: Math.max(0, Math.min(entries.length - 1, currentIndex)),
  entries: entries.map((entry) => ({
    ...entry,
    adjustments: cloneAdjustments(entry.adjustments),
  })),
});

export const appendHistoryEntry = (
  entries: EditHistoryEntry[],
  currentIndex: number,
  adjustments: Adjustments,
  label: string,
): EditHistoryState => {
  const currentEntry = entries[currentIndex];
  if (currentEntry && areAdjustmentsEqual(currentEntry.adjustments, adjustments)) {
    return { version: EDIT_HISTORY_VERSION, currentIndex, entries };
  }

  const nextEntries = [...entries.slice(0, currentIndex + 1), createHistoryEntry(label, adjustments)];
  const limitedEntries = nextEntries.slice(Math.max(0, nextEntries.length - EDIT_HISTORY_LIMIT));
  return {
    version: EDIT_HISTORY_VERSION,
    currentIndex: limitedEntries.length - 1,
    entries: limitedEntries,
  };
};

export const inferHistoryLabel = (previous: Adjustments, next: Adjustments): string => {
  const changedKeys = Object.keys(next).filter((key) => !areAdjustmentsEqual(previous?.[key], next?.[key]));
  if (changedKeys.length === 0) return 'Edit';
  if (changedKeys.every((key) => key === 'sectionVisibility' || key === 'showClipping')) return 'View Options';
  if (changedKeys.some((key) => GEOMETRY_KEYS.has(key))) return 'Crop & Transform';
  if (changedKeys.length === 1) return LABEL_BY_KEY[changedKeys[0]] || 'Edit';

  const labels = Array.from(new Set(changedKeys.map((key) => LABEL_BY_KEY[key]).filter(Boolean)));
  if (labels.length === 1) return labels[0];
  if (changedKeys.some((key) => key === 'aiPatches')) return 'AI Edit';
  if (changedKeys.some((key) => key === 'masks')) return 'Masking';
  return 'Adjustments';
};
