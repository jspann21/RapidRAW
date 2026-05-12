import { useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useShallow } from 'zustand/react/shallow';
import { useSettingsStore } from '../store/useSettingsStore';
import { useUIStore } from '../store/useUIStore';
import { useLibraryStore } from '../store/useLibraryStore';
import { useEditorStore } from '../store/useEditorStore';
import { useGooglePhotosStore } from '../store/useGooglePhotosStore';
import { THEMES, DEFAULT_THEME_ID, ThemeProps } from '../utils/themes';
import { COPYABLE_ADJUSTMENT_KEYS } from '../utils/adjustments';
import {
  FilterCriteria,
  Invokes,
  LibraryViewMode,
  RawStatus,
  Theme,
  ThumbnailSize,
  ThumbnailAspectRatio,
} from '../components/ui/AppProperties';

interface UseAppInitializationProps {
  preloadedDataRef: React.RefObject<any>;
  thumbnailSize: ThumbnailSize;
  setThumbnailSize: (size: ThumbnailSize) => void;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  setThumbnailAspectRatio: (ratio: ThumbnailAspectRatio) => void;
  libraryViewMode: LibraryViewMode;
  setLibraryViewMode: (mode: LibraryViewMode) => void;
}

export const useAppInitialization = ({
  preloadedDataRef,
  thumbnailSize,
  setThumbnailSize,
  thumbnailAspectRatio,
  setThumbnailAspectRatio,
  libraryViewMode,
  setLibraryViewMode,
}: UseAppInitializationProps) => {
  const isInitialMount = useRef(true);

  const {
    appSettings,
    theme,
    osPlatform,
    setAppSettings,
    setTheme,
    setSupportedTypes,
    initPlatform,
    handleSettingsChange,
  } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      theme: state.theme,
      osPlatform: state.osPlatform,
      setAppSettings: state.setAppSettings,
      setTheme: state.setTheme,
      setSupportedTypes: state.setSupportedTypes,
      initPlatform: state.initPlatform,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const { uiVisibility, setUI } = useUIStore(
    useShallow((state) => ({
      uiVisibility: state.uiVisibility,
      setUI: state.setUI,
    })),
  );

  const { sortCriteria, filterCriteria, setSortCriteria, setFilterCriteria, setLibrary } = useLibraryStore(
    useShallow((state) => ({
      sortCriteria: state.sortCriteria,
      filterCriteria: state.filterCriteria,
      setSortCriteria: state.setSortCriteria,
      setFilterCriteria: state.setFilterCriteria,
      setLibrary: state.setLibrary,
    })),
  );

  const { setEditor } = useEditorStore(
    useShallow((state) => ({
      setEditor: state.setEditor,
    })),
  );

  const isAndroid = osPlatform === 'android';
  const defaultThumbnailSize = isAndroid ? ThumbnailSize.Small : ThumbnailSize.Medium;
  const defaultLibraryViewMode = isAndroid ? LibraryViewMode.Recursive : LibraryViewMode.Flat;

  useEffect(() => {
    initPlatform();
  }, [initPlatform]);

  useEffect(() => {
    invoke(Invokes.GetSupportedFileTypes)
      .then((types: any) => setSupportedTypes(types))
      .catch((err) => console.error('Failed to load supported file types:', err));
  }, [setSupportedTypes]);

  useEffect(() => {
    invoke(Invokes.LoadSettings)
      .then(async (settings: any) => {
        if (
          !settings.copyPasteSettings ||
          !settings.copyPasteSettings.includedAdjustments ||
          settings.copyPasteSettings.includedAdjustments.length === 0
        ) {
          settings.copyPasteSettings = { mode: 'merge', includedAdjustments: COPYABLE_ADJUSTMENT_KEYS };
        }
        setAppSettings(settings);

        if (settings?.sortCriteria) setSortCriteria(settings.sortCriteria);

        if (settings?.filterCriteria) {
          setFilterCriteria((prev: FilterCriteria) => ({
            ...prev,
            ...settings.filterCriteria,
            rawStatus: settings.filterCriteria.rawStatus || RawStatus.All,
            colors: settings.filterCriteria.colors || [],
          }));
        }

        if (settings?.theme) setTheme(settings.theme);

        if (settings?.uiVisibility)
          setUI((state) => ({ uiVisibility: { ...state.uiVisibility, ...settings.uiVisibility } }));

        if (settings?.isWaveformVisible !== undefined) setEditor({ isWaveformVisible: settings.isWaveformVisible });
        if (settings?.activeWaveformChannel) setEditor({ activeWaveformChannel: settings.activeWaveformChannel });
        if (typeof settings?.waveformHeight === 'number') setEditor({ waveformHeight: settings.waveformHeight });

        setLibraryViewMode(settings?.libraryViewMode ?? defaultLibraryViewMode);
        setThumbnailSize(settings?.thumbnailSize ?? defaultThumbnailSize);
        if (settings?.thumbnailAspectRatio) setThumbnailAspectRatio(settings.thumbnailAspectRatio);

        if (settings?.googlePhotosIntegrationEnabled) {
          invoke(Invokes.GooglePhotosGetSyncIndex)
            .then((index: any) => useGooglePhotosStore.getState().setSyncedEntries(index || {}))
            .catch((err) => console.error('Failed to load Google Photos sync index:', err));
        }

        if (settings?.pinnedFolders && settings.pinnedFolders.length > 0) {
          try {
            const trees = await invoke(Invokes.GetPinnedFolderTrees, {
              paths: settings.pinnedFolders,
              expandedFolders: settings.lastFolderState?.expandedFolders || [],
              showImageCounts: settings.enableFolderImageCounts ?? false,
            });
            setLibrary({ pinnedFolderTrees: trees });
          } catch (err) {
            console.error('Failed to load pinned folder trees:', err);
          }
        }

        if (!isAndroid && settings.lastRootPath) {
          const root = settings.lastRootPath;
          const currentPath = settings.lastFolderState?.currentFolderPath || root;
          const command =
            settings.libraryViewMode === LibraryViewMode.Recursive
              ? Invokes.ListImagesRecursive
              : Invokes.ListImagesInDir;

          preloadedDataRef.current = {
            rootPath: root,
            currentPath: currentPath,
            tree: invoke(Invokes.GetFolderTree, {
              path: root,
              expandedFolders: settings.lastFolderState?.expandedFolders ?? [root],
              showImageCounts: settings.enableFolderImageCounts ?? false,
            }),
            images: invoke(command, { path: currentPath }),
          };
        }

        invoke('frontend_ready').catch((e) => console.error('Failed to notify backend of readiness:', e));
      })
      .catch((err) => {
        console.error('Failed to load settings:', err);
        setAppSettings({
          lastRootPath: null,
          theme: DEFAULT_THEME_ID as Theme,
          thumbnailSize: defaultThumbnailSize,
          libraryViewMode: defaultLibraryViewMode,
        });
      })
      .finally(() => {
        isInitialMount.current = false;
      });
  }, [
    isAndroid,
    setAppSettings,
    setTheme,
    setUI,
    defaultLibraryViewMode,
    defaultThumbnailSize,
    setSortCriteria,
    setFilterCriteria,
    setEditor,
    setLibrary,
    preloadedDataRef,
    setLibraryViewMode,
    setThumbnailSize,
    setThumbnailAspectRatio,
  ]);

  // 4. Settings Synchronization Effects
  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (JSON.stringify(appSettings.uiVisibility) !== JSON.stringify(uiVisibility)) {
      handleSettingsChange({ ...appSettings, uiVisibility });
    }
  }, [uiVisibility, appSettings, handleSettingsChange]);

  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (appSettings.thumbnailSize !== thumbnailSize) {
      handleSettingsChange({ ...appSettings, thumbnailSize });
    }
  }, [thumbnailSize, appSettings, handleSettingsChange]);

  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (appSettings.thumbnailAspectRatio !== thumbnailAspectRatio) {
      handleSettingsChange({ ...appSettings, thumbnailAspectRatio });
    }
  }, [thumbnailAspectRatio, appSettings, handleSettingsChange]);

  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (appSettings.libraryViewMode !== libraryViewMode) {
      handleSettingsChange({ ...appSettings, libraryViewMode });
    }
  }, [libraryViewMode, appSettings, handleSettingsChange]);

  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (JSON.stringify(appSettings.sortCriteria) !== JSON.stringify(sortCriteria)) {
      handleSettingsChange({ ...appSettings, sortCriteria });
    }
  }, [sortCriteria, appSettings, handleSettingsChange]);

  useEffect(() => {
    if (isInitialMount.current || !appSettings) return;
    if (JSON.stringify(appSettings.filterCriteria) !== JSON.stringify(filterCriteria)) {
      handleSettingsChange({ ...appSettings, filterCriteria });
    }
  }, [filterCriteria, appSettings, handleSettingsChange]);

  useEffect(() => {
    const root = document.documentElement;
    const currentThemeId = theme || DEFAULT_THEME_ID;

    const baseTheme =
      THEMES.find((t: ThemeProps) => t.id === currentThemeId) ||
      THEMES.find((t: ThemeProps) => t.id === DEFAULT_THEME_ID);
    if (!baseTheme) return;

    let finalCssVariables: any = { ...baseTheme.cssVariables };

    Object.entries(finalCssVariables).forEach(([key, value]) => {
      root.style.setProperty(key, value as string);
    });

    const fontFamily = appSettings?.fontFamily || 'poppins';
    const fontStack =
      fontFamily === 'system'
        ? '-apple-system, BlinkMacSystemFont, system-ui, sans-serif'
        : "'Poppins', system-ui, sans-serif";
    root.style.setProperty('--font-family', fontStack);
  }, [theme, appSettings?.fontFamily]);
};
