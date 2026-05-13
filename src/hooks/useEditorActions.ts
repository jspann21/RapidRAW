import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import debounce from 'lodash.debounce';
import { toast } from 'react-toastify';
import { useEditorStore } from '../store/useEditorStore';
import { useLibraryStore } from '../store/useLibraryStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { useProcessStore } from '../store/useProcessStore';
import { PasteAdjustmentsUndoSnapshot } from '../store/useEditorStore';
import {
  Adjustments,
  INITIAL_ADJUSTMENTS,
  COPYABLE_ADJUSTMENT_KEYS,
  PasteMode,
  normalizeLoadedAdjustments,
} from '../utils/adjustments';
import { calculateCenteredCrop } from '../utils/cropUtils';
import { Invokes } from '../components/ui/AppProperties';
import { globalImageCache } from '../utils/ImageLRUCache';
import { inferHistoryLabel, serializeEditHistory } from '../utils/editHistory';

export const debouncedSave = debounce((path: string, adjustmentsToSave: Adjustments) => {
  const { history, historyIndex } = useEditorStore.getState();
  const editHistory = serializeEditHistory(history, historyIndex);
  invoke(Invokes.SaveMetadataAndUpdateThumbnail, { path, adjustments: adjustmentsToSave, editHistory }).catch((err) => {
    console.error('Auto-save failed:', err);
    toast.error(`Failed to save changes: ${err}`);
  });
}, 300);

export const debouncedSetHistory = debounce((newAdj: Adjustments, label?: string) => {
  const { selectedImage, pushHistory } = useEditorStore.getState();
  pushHistory(newAdj, label);
  if (selectedImage?.path) {
    debouncedSave(selectedImage.path, newAdj);
  }
}, 500);

const cloneCopyableAdjustments = (sourceAdjustments: Adjustments): Partial<Adjustments> => {
  const adjustmentsToCopy: Partial<Adjustments> = {};

  for (const key of COPYABLE_ADJUSTMENT_KEYS) {
    if (Object.prototype.hasOwnProperty.call(sourceAdjustments, key)) {
      adjustmentsToCopy[key as keyof Adjustments] = structuredClone(sourceAdjustments[key as keyof Adjustments]);
    }
  }

  return adjustmentsToCopy;
};

const hasAdjustmentPayload = (value: unknown): value is Adjustments => {
  if (!value) return false;
  if (typeof value !== 'object') return false;
  return !('is_null' in value && (value as { is_null?: boolean }).is_null);
};

const toNormalizedAdjustments = (adjustments: unknown): Adjustments => {
  return hasAdjustmentPayload(adjustments) ? normalizeLoadedAdjustments(adjustments) : { ...INITIAL_ADJUSTMENTS };
};

const loadPasteUndoSnapshot = async (
  path: string,
  activeImagePath: string | null | undefined,
  activeAdjustments: Adjustments,
): Promise<PasteAdjustmentsUndoSnapshot> => {
  if (path === activeImagePath) {
    const currentAdjustments = structuredClone(activeAdjustments);
    return {
      path,
      adjustments: currentAdjustments,
      normalizedAdjustments: currentAdjustments,
    };
  }

  const metadata = await invoke<{ adjustments?: unknown }>(Invokes.LoadMetadata, { path });
  const savedAdjustments = hasAdjustmentPayload(metadata.adjustments) ? structuredClone(metadata.adjustments) : null;

  return {
    path,
    adjustments: savedAdjustments,
    normalizedAdjustments: toNormalizedAdjustments(savedAdjustments),
  };
};

const saveAdjustmentsForPath = (path: string, adjustments: unknown, editHistory?: unknown) => {
  return invoke(Invokes.SaveMetadataAndUpdateThumbnail, { path, adjustments, editHistory });
};

export function useEditorActions() {
  const setEditor = useEditorStore((s) => s.setEditor);

  const setAdjustments = useCallback(
    (value: Partial<Adjustments> | ((prev: Adjustments) => Adjustments), historyLabel?: string) => {
      setEditor((state) => {
        const prev = state.adjustments;
        const newAdjustments = typeof value === 'function' ? value(prev) : { ...prev, ...value };
        debouncedSetHistory(newAdjustments, historyLabel || inferHistoryLabel(prev, newAdjustments));
        return { adjustments: newAdjustments };
      });
    },
    [setEditor],
  );

  const handleRotate = useCallback(
    (degrees: number) => {
      const { selectedImage, adjustments } = useEditorStore.getState();
      const increment = degrees > 0 ? 1 : 3;
      const newAspectRatio =
        adjustments.aspectRatio && adjustments.aspectRatio !== 0 ? 1 / adjustments.aspectRatio : null;
      const newOrientationSteps = ((adjustments.orientationSteps || 0) + increment) % 4;
      const newCrop =
        selectedImage?.width && selectedImage?.height
          ? calculateCenteredCrop(selectedImage.width, selectedImage.height, newOrientationSteps, newAspectRatio)
          : null;

      setAdjustments(
        (prev) => ({
          ...prev,
          aspectRatio: newAspectRatio,
          orientationSteps: newOrientationSteps,
          rotation: 0,
          crop: newCrop,
        }),
        degrees > 0 ? 'Rotate Right' : 'Rotate Left',
      );
    },
    [setAdjustments],
  );

  const handleAutoAdjustments = useCallback(async () => {
    const selectedImage = useEditorStore.getState().selectedImage;
    if (!selectedImage?.isReady) return;
    try {
      const autoAdjustments: Adjustments = await invoke(Invokes.CalculateAutoAdjustments);
      setAdjustments(
        (prev: Adjustments) => ({
          ...prev,
          ...autoAdjustments,
          sectionVisibility: { ...prev.sectionVisibility, ...autoAdjustments.sectionVisibility },
        }),
        'Auto Adjustment',
      );
    } catch (err) {
      toast.error(`Failed to apply auto adjustments: ${err}`);
    }
  }, [setAdjustments]);

  const handleLutSelect = useCallback(
    async (path: string) => {
      const isAndroid = useSettingsStore.getState().osPlatform === 'android';
      try {
        const result: { size: number } = await invoke('load_and_parse_lut', { path });
        const name = isAndroid
          ? await invoke<string>('resolve_android_content_uri_name', { uriStr: path })
          : path.split(/[\\/]/).pop() || 'LUT';
        setAdjustments(
          (prev: Adjustments) => ({
            ...prev,
            lutPath: path,
            lutName: name,
            lutSize: result.size,
            lutIntensity: 100,
            sectionVisibility: { ...(prev.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility), effects: true },
          }),
          'LUT',
        );
      } catch (err) {
        toast.error(`Failed to load LUT: ${err}`);
      }
    },
    [setAdjustments],
  );

  const handleResetAdjustments = useCallback(
    (paths?: string[]) => {
      const { multiSelectedPaths, libraryActivePath, setLibrary } = useLibraryStore.getState();
      const { selectedImage, resetHistory } = useEditorStore.getState();
      const pathsToReset = paths || multiSelectedPaths;
      if (pathsToReset.length === 0) return;

      pathsToReset.forEach((p) => globalImageCache.delete(p));
      debouncedSetHistory.cancel();

      invoke(Invokes.ResetAdjustmentsForPaths, { paths: pathsToReset })
        .then(() => {
          if (libraryActivePath && pathsToReset.includes(libraryActivePath))
            setLibrary({ libraryActiveAdjustments: { ...INITIAL_ADJUSTMENTS } });
          if (selectedImage && pathsToReset.includes(selectedImage.path)) {
            const aspect =
              selectedImage.width && selectedImage.height ? selectedImage.width / selectedImage.height : null;
            const resetData = { ...INITIAL_ADJUSTMENTS, aspectRatio: aspect, aiPatches: [] };
            resetHistory(resetData);
            setEditor({ adjustments: resetData });
            useEditorStore.getState().pushHistory(resetData, 'Reset Adjustments');
          }
        })
        .catch((err) => toast.error(`Failed to reset adjustments: ${err}`));
    },
    [setEditor],
  );

  const handleCopyAdjustments = useCallback(async (sourcePath?: string) => {
    const { selectedImage, adjustments } = useEditorStore.getState();
    const { libraryActivePath, multiSelectedPaths, setLibrary } = useLibraryStore.getState();
    const targetPath =
      sourcePath ??
      selectedImage?.path ??
      libraryActivePath ??
      (multiSelectedPaths.length === 1 ? multiSelectedPaths[0] : null);

    try {
      let sourceAdjustments: Adjustments;

      if (selectedImage && targetPath === selectedImage.path) {
        sourceAdjustments = adjustments;
      } else if (targetPath) {
        const metadata = await invoke<{ adjustments?: unknown }>(Invokes.LoadMetadata, { path: targetPath });
        sourceAdjustments = hasAdjustmentPayload(metadata.adjustments)
          ? normalizeLoadedAdjustments(metadata.adjustments)
          : { ...INITIAL_ADJUSTMENTS };
        setLibrary({ libraryActiveAdjustments: sourceAdjustments });
      } else {
        sourceAdjustments = useLibraryStore.getState().libraryActiveAdjustments;
      }

      useEditorStore
        .getState()
        .setEditor({ copiedAdjustments: cloneCopyableAdjustments(sourceAdjustments) as Adjustments });
      useProcessStore.getState().setProcess({ isCopied: true });
    } catch (err) {
      toast.error(`Failed to copy adjustments: ${err}`);
    }
  }, []);

  const handlePasteAdjustments = useCallback(
    async (paths?: string[]) => {
      const { copiedAdjustments, selectedImage, adjustments } = useEditorStore.getState();
      const { multiSelectedPaths, libraryActivePath, setLibrary } = useLibraryStore.getState();
      const { appSettings } = useSettingsStore.getState();
      const { setProcess } = useProcessStore.getState();

      if (!copiedAdjustments || !appSettings) return;

      const { mode, includedAdjustments } = appSettings.copyPasteSettings;
      const adjustmentsToApply: Partial<Adjustments> = {};

      for (const key of includedAdjustments) {
        if (Object.prototype.hasOwnProperty.call(copiedAdjustments, key)) {
          const value = copiedAdjustments[key as keyof Adjustments];
          if (mode === PasteMode.Merge) {
            const defaultValue = INITIAL_ADJUSTMENTS[key as keyof Adjustments];
            if (JSON.stringify(value) !== JSON.stringify(defaultValue))
              adjustmentsToApply[key as keyof Adjustments] = value;
          } else {
            adjustmentsToApply[key as keyof Adjustments] = value;
          }
        }
      }

      if (Object.keys(adjustmentsToApply).length === 0) {
        setProcess({ isPasted: true });
        return;
      }

      const pathsToUpdate = Array.from(
        new Set(
          paths || (multiSelectedPaths.length > 0 ? multiSelectedPaths : selectedImage ? [selectedImage.path] : []),
        ),
      );
      if (pathsToUpdate.length === 0) return;

      let undoSnapshots: PasteAdjustmentsUndoSnapshot[];
      try {
        undoSnapshots = await Promise.all(
          pathsToUpdate.map((path) => loadPasteUndoSnapshot(path, selectedImage?.path, adjustments)),
        );
      } catch (err) {
        toast.error(`Failed to prepare paste undo: ${err}`);
        return;
      }

      const nextAdjustmentsByPath = undoSnapshots.map((snapshot) => ({
        path: snapshot.path,
        adjustments: structuredClone({ ...snapshot.normalizedAdjustments, ...adjustmentsToApply }) as Adjustments,
      }));

      if (selectedImage && pathsToUpdate.includes(selectedImage.path)) {
        const nextActiveAdjustments = nextAdjustmentsByPath.find(
          (item) => item.path === selectedImage.path,
        )?.adjustments;
        if (nextActiveAdjustments) {
          setAdjustments(nextActiveAdjustments, 'Paste Adjustments');
        }
      }

      useEditorStore.getState().setEditor((state) => ({
        pasteAdjustmentsUndoStack: [...state.pasteAdjustmentsUndoStack, { snapshots: undoSnapshots }].slice(-20),
      }));

      pathsToUpdate.forEach((p) => globalImageCache.delete(p));

      try {
        await Promise.all(
          nextAdjustmentsByPath.map(({ path, adjustments }) => saveAdjustmentsForPath(path, adjustments)),
        );
        const nextLibraryActive = libraryActivePath
          ? nextAdjustmentsByPath.find((item) => item.path === libraryActivePath)
          : null;
        if (nextLibraryActive) {
          setLibrary({ libraryActiveAdjustments: nextLibraryActive.adjustments });
        }
      } catch (err) {
        toast.error(`Failed to paste adjustments: ${err}`);
      }

      setProcess({ isPasted: true });
    },
    [setAdjustments],
  );

  const handleUndoPasteAdjustments = useCallback(async () => {
    const { pasteAdjustmentsUndoStack } = useEditorStore.getState();
    const undoEntry = pasteAdjustmentsUndoStack[pasteAdjustmentsUndoStack.length - 1];
    if (!undoEntry) return false;

    debouncedSetHistory.cancel();
    debouncedSave.cancel();

    try {
      undoEntry.snapshots.forEach((snapshot) => globalImageCache.delete(snapshot.path));
      await Promise.all(
        undoEntry.snapshots.map((snapshot) => saveAdjustmentsForPath(snapshot.path, snapshot.adjustments)),
      );

      const { selectedImage, resetHistory, setEditor } = useEditorStore.getState();
      const { libraryActivePath, setLibrary } = useLibraryStore.getState();
      const activeSnapshot = selectedImage
        ? undoEntry.snapshots.find((snapshot) => snapshot.path === selectedImage.path)
        : null;
      const libraryActiveSnapshot = libraryActivePath
        ? undoEntry.snapshots.find((snapshot) => snapshot.path === libraryActivePath)
        : null;

      setEditor((state) => ({
        pasteAdjustmentsUndoStack: state.pasteAdjustmentsUndoStack.slice(0, -1),
      }));

      if (activeSnapshot) {
        setEditor({ suppressNextMultiSelectionSync: true });
        resetHistory(structuredClone(activeSnapshot.normalizedAdjustments));
        useEditorStore.getState().pushHistory(structuredClone(activeSnapshot.normalizedAdjustments), 'Undo Paste');
      }

      if (libraryActiveSnapshot) {
        setLibrary({ libraryActiveAdjustments: structuredClone(libraryActiveSnapshot.normalizedAdjustments) });
      }

      useProcessStore.getState().setProcess({ isPasted: true });
      return true;
    } catch (err) {
      toast.error(`Failed to undo pasted adjustments: ${err}`);
      return false;
    }
  }, []);

  const handleZoomChange = useCallback((zoomValue: number, fitToWindow: boolean = false) => {
    const { originalSize, baseRenderSize, adjustments } = useEditorStore.getState();
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    let targetZoomPercent: number;

    const orientationSteps = adjustments.orientationSteps || 0;
    const isSwapped = orientationSteps === 1 || orientationSteps === 3;
    const effectiveOriginalWidth = isSwapped ? originalSize.height : originalSize.width;
    const effectiveOriginalHeight = isSwapped ? originalSize.width : originalSize.height;

    if (fitToWindow) {
      if (
        effectiveOriginalWidth > 0 &&
        effectiveOriginalHeight > 0 &&
        baseRenderSize.width > 0 &&
        baseRenderSize.height > 0
      ) {
        const originalAspect = effectiveOriginalWidth / effectiveOriginalHeight;
        const baseAspect = baseRenderSize.width / baseRenderSize.height;
        targetZoomPercent =
          originalAspect > baseAspect
            ? baseRenderSize.width / effectiveOriginalWidth
            : baseRenderSize.height / effectiveOriginalHeight;
      } else {
        targetZoomPercent = 1.0;
      }
    } else {
      targetZoomPercent = zoomValue / dpr;
    }

    targetZoomPercent = Math.max(0.1 / dpr, Math.min(2.0, targetZoomPercent));

    let transformZoom = 1.0;
    if (
      effectiveOriginalWidth > 0 &&
      effectiveOriginalHeight > 0 &&
      baseRenderSize.width > 0 &&
      baseRenderSize.height > 0
    ) {
      const originalAspect = effectiveOriginalWidth / effectiveOriginalHeight;
      const baseAspect = baseRenderSize.width / baseRenderSize.height;
      if (originalAspect > baseAspect) {
        transformZoom = (targetZoomPercent * effectiveOriginalWidth) / baseRenderSize.width;
      } else {
        transformZoom = (targetZoomPercent * effectiveOriginalHeight) / baseRenderSize.height;
      }
    }
    useEditorStore.getState().setEditor({ zoom: transformZoom });
  }, []);

  return {
    setAdjustments,
    handleRotate,
    handleAutoAdjustments,
    handleLutSelect,
    handleResetAdjustments,
    handleCopyAdjustments,
    handlePasteAdjustments,
    handleUndoPasteAdjustments,
    handleZoomChange,
  };
}
