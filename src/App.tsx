import { type PointerEvent as ReactPointerEvent, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ClerkProvider } from '@clerk/react';
import { ToastContainer, toast, Slide } from 'react-toastify';
import { DndContext, DragOverlay, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import clsx from 'clsx';

import TitleBar from './window/TitleBar';
import FolderTree from './components/panel/right/FolderTree';
import SettingsPanel from './components/panel/SettingsPanel';
import ExportPanel from './components/panel/right/ExportPanel';
import GlobalTooltip from './components/ui/GlobalTooltip';
import AppModals from './components/modals/AppModals';

import SidePanelArea from './components/panel/SidePanelArea';
import { PANEL_ICONS } from './components/panel/PanelSwitcher';
import Controls from './components/panel/right/ControlsPanel';
import MetadataPanel from './components/panel/right/MetadataPanel';
import CropPanel from './components/panel/right/CropPanel';
import MasksPanel from './components/panel/right/MasksPanel';
import AIPanel from './components/panel/right/AIPanel';
import PresetsPanel from './components/panel/right/PresetsPanel';
import HistoryPanel from './components/panel/right/HistoryPanel';

import EditorView from './components/views/EditorView';
import LibraryView from './components/views/LibraryView';

import { ContextMenuProvider } from './context/ContextMenuContext';
import { useSettingsStore } from './store/useSettingsStore';
import { useUIStore } from './store/useUIStore';
import { useLibraryStore } from './store/useLibraryStore';
import { useEditorStore } from './store/useEditorStore';
import { useProcessStore } from './store/useProcessStore';
import { GOOGLE_PHOTOS_FOLDER_PATH } from './store/useGooglePhotosStore';
import { useShallow } from 'zustand/react/shallow';

import { useThumbnails } from './hooks/useThumbnails';
import { ImageDimensions } from './hooks/useImageRenderSize';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useTauriListeners } from './hooks/useTauriListeners';
import { useFileOperations } from './hooks/useFileOperations';
import { useAppContextMenus } from './hooks/useAppContextMenus';
import { useSortedLibrary } from './hooks/useSortedLibrary';
import { useAppNavigation } from './hooks/useAppNavigation';
import { useExternalEditSession } from './hooks/useExternalEditSession';
import ExternalEditBar from './components/ui/ExternalEditBar';
import { Status } from './components/ui/ExportImportProperties';

import { useEditorActions } from './hooks/useEditorActions';
import { useLibraryActions } from './hooks/useLibraryActions';
import { useProductivityActions } from './hooks/useProductivityActions';

import { normalizeDraggedImagePaths } from './utils/imageDragDrop';
import { useAppInitialization } from './hooks/useAppInitialization';
import { useAndroidBackHandler } from './hooks/useAndroidBackHandler';
import './i18n';

import {
  Invokes,
  ImageFile,
  LibraryViewMode,
  Panel,
  PanelRegion,
  Theme,
  ThumbnailSize,
  ThumbnailAspectRatio,
} from './components/ui/AppProperties';

import ImageProcessingManager from './components/managers/ImageProcessingManager';
import ImageLoaderManager from './components/managers/ImageLoaderManager';

const CLERK_PUBLISHABLE_KEY = 'pk_test_YnJpZWYtc2Vhc25haWwtMTIuY2xlcmsuYWNjb3VudHMuZGV2JA'; // local dev key

const insertChildrenIntoTree = (node: any, targetPath: string, newChildren: any[]): any => {
  if (!node) return null;

  if (node.path === targetPath) {
    const mergedChildren = newChildren.map((newChild: any) => {
      const existingChild = node.children?.find((c: any) => c.path === newChild.path);
      if (existingChild && existingChild.children && existingChild.children.length > 0) {
        return { ...newChild, children: existingChild.children };
      }
      return newChild;
    });
    return { ...node, children: mergedChildren };
  }

  if (node.children && node.children.length > 0) {
    return {
      ...node,
      children: node.children.map((child: any) => insertChildrenIntoTree(child, targetPath, newChildren)),
    };
  }

  return node;
};

const getFolderDropTargetPath = (target: EventTarget | null) => {
  if (!(target instanceof Element)) return null;
  return target.closest<HTMLElement>('[data-folder-path]')?.dataset.folderPath || null;
};

const isFolderSidebarTarget = (target: EventTarget | Element | null) => {
  return target instanceof Element && !!target.closest('[data-folder-sidebar]');
};

const getLibraryImageDragSourcePath = (target: EventTarget | null) => {
  if (!(target instanceof Element)) return null;
  return target.closest<HTMLElement>('[data-library-image-path]')?.dataset.libraryImagePath || null;
};

interface LibraryImagePointerDrag {
  dragging: boolean;
  paths: string[];
  pointerId: number;
  sourcePath: string;
  startX: number;
  startY: number;
}

function App() {
  const COMPACT_EDITOR_MAX_WIDTH = 900;

  const { appSettings, theme, osPlatform, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      theme: state.theme,
      osPlatform: state.osPlatform,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const {
    activeView,
    isFullScreen,
    isWindowFullScreen,
    isInstantTransition,
    isLayoutReady,
    uiVisibility,
    isLibraryExportPanelVisible,
    leftPanelWidth,
    rightPanelWidth,
    compactEditorPanelHeightOverride,
    activeRightPanel,
    activeLayoutDragItem,
    isSettingsOpen,
    settingsPanelRequest,
    setUI,
    setRightPanel,
    setLayoutDragItem,
    movePanel,
  } = useUIStore(
    useShallow((state) => ({
      activeView: state.activeView,
      isFullScreen: state.isFullScreen,
      isWindowFullScreen: state.isWindowFullScreen,
      isInstantTransition: state.isInstantTransition,
      isLayoutReady: state.isLayoutReady,
      uiVisibility: state.uiVisibility,
      isLibraryExportPanelVisible: state.isLibraryExportPanelVisible,
      leftPanelWidth: state.leftPanelWidth,
      rightPanelWidth: state.rightPanelWidth,
      compactEditorPanelHeightOverride: state.compactEditorPanelHeightOverride,
      activeRightPanel: state.activeRightPanel,
      activeLayoutDragItem: state.activeLayoutDragItem,
      isSettingsOpen: state.isSettingsOpen,
      settingsPanelRequest: state.settingsPanelRequest,
      setUI: state.setUI,
      setRightPanel: state.setRightPanel,
      setLayoutDragItem: state.setLayoutDragItem,
      movePanel: state.movePanel,
    })),
  );

  const { rootPaths, currentFolderPath, expandedFolders, multiSelectedPaths, setLibrary } = useLibraryStore(
    useShallow((state) => ({
      rootPaths: state.rootPaths,
      currentFolderPath: state.currentFolderPath,
      expandedFolders: state.expandedFolders,
      multiSelectedPaths: state.multiSelectedPaths,
      setLibrary: state.setLibrary,
    })),
  );

  const { selectedImage, activeMaskContainerId, activeAiPatchContainerId, hasRenderedFirstFrame, setEditor } =
    useEditorStore(
      useShallow((state) => ({
        selectedImage: state.selectedImage,
        activeMaskContainerId: state.activeMaskContainerId,
        activeAiPatchContainerId: state.activeAiPatchContainerId,
        hasRenderedFirstFrame: state.hasRenderedFirstFrame,
        setEditor: state.setEditor,
      })),
    );

  const { exportState, setExportState } = useProcessStore(
    useShallow((state) => ({
      exportState: state.exportState,
      setExportState: state.setExportState,
    })),
  );

  const defaultThumbnailSize = osPlatform === 'android' ? ThumbnailSize.Small : ThumbnailSize.Medium;
  const defaultLibraryViewMode = osPlatform === 'android' ? LibraryViewMode.Recursive : LibraryViewMode.Flat;

  const selectedImagePathRef = useRef<string | null>(null);
  const libraryImagePointerDragRef = useRef<LibraryImagePointerDrag | null>(null);
  const suppressNextClickRef = useRef(false);
  const [libraryDragPreview, setLibraryDragPreview] = useState<{ count: number; x: number; y: number } | null>(null);
  useEffect(() => {
    selectedImagePathRef.current = selectedImage?.path ?? null;
  }, [selectedImage?.path]);

  const prevAdjustmentsRef = useRef<any>(null);

  const [viewportSize, setViewportSize] = useState<ImageDimensions>(() => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0 };
    }

    return {
      width: Math.round(window.visualViewport?.width ?? window.innerWidth),
      height: Math.round(window.visualViewport?.height ?? window.innerHeight),
    };
  });

  const isBackendReadyRef = useRef(true);
  const previewJobIdRef = useRef<number>(0);
  const latestRenderedJobIdRef = useRef<number>(0);
  const currentResRef = useRef<number>(1280);
  const cachedEditStateRef = useRef<any | null>(null);

  const [libraryViewMode, setLibraryViewMode] = useState<LibraryViewMode>(defaultLibraryViewMode);
  const [isResizing, setIsResizing] = useState(false);
  const [thumbnailSize, setThumbnailSize] = useState(defaultThumbnailSize);
  const [thumbnailAspectRatio, setThumbnailAspectRatio] = useState(ThumbnailAspectRatio.Cover);

  const { requestThumbnails, clearThumbnailQueue, markGenerated } = useThumbnails();

  const transformWrapperRef = useRef<any>(null);
  const preloadedDataRef = useRef<{
    trees?: Promise<any>;
    images?: Promise<ImageFile[]>;
    rootPaths?: string[];
    currentPath?: string;
  }>({});

  useAppInitialization({
    preloadedDataRef,
    thumbnailSize,
    setThumbnailSize,
    thumbnailAspectRatio,
    setThumbnailAspectRatio,
    libraryViewMode,
    setLibraryViewMode,
  });

  const isAndroid = osPlatform === 'android';
  const isPortraitViewport = viewportSize.width > 0 && viewportSize.height > viewportSize.width;
  const isCompactPortrait =
    viewportSize.width > 0 && viewportSize.width <= COMPACT_EDITOR_MAX_WIDTH && isPortraitViewport;

  const compactEditorPanelMinHeight = 220;
  const compactEditorPanelMaxHeight =
    viewportSize.height > 0
      ? Math.max(compactEditorPanelMinHeight, Math.min(Math.round(viewportSize.height * 0.85), 850))
      : 520;

  const getDynamicCompactPanelHeight = () => {
    const { originalSize, adjustments } = useEditorStore.getState();
    const halfScreenHeight = viewportSize.height > 0 ? Math.round(viewportSize.height * 0.5) : 340;

    if (!selectedImage || originalSize.width === 0 || originalSize.height === 0 || viewportSize.width === 0) {
      return halfScreenHeight;
    }
    let effectiveRatio = originalSize.width / originalSize.height;
    const orientationSteps = adjustments?.orientationSteps || 0;
    if (orientationSteps % 2 !== 0) {
      effectiveRatio = originalSize.height / originalSize.width;
    }
    if (adjustments?.aspectRatio && adjustments.aspectRatio > 0) {
      effectiveRatio = adjustments.aspectRatio;
    }
    const desiredImageHeight = viewportSize.width / effectiveRatio;
    const topUiEstimation = !appSettings?.decorations && !isWindowFullScreen ? 110 : 60;
    const totalDesiredTopHeight = desiredImageHeight + topUiEstimation;
    const calculatedBottomHeight = Math.round(viewportSize.height - totalDesiredTopHeight);
    return Math.max(halfScreenHeight, calculatedBottomHeight);
  };

  const compactEditorPanelDefaultHeight = getDynamicCompactPanelHeight();
  const compactEditorPanelHeight = Math.max(
    compactEditorPanelMinHeight,
    Math.min(compactEditorPanelHeightOverride ?? compactEditorPanelDefaultHeight, compactEditorPanelMaxHeight),
  );
  const compactEditorPanelCollapsedHeight = 96;

  const { handleCopyAdjustments, handlePasteAdjustments, handleResetAdjustments, handleZoomChange } =
    useEditorActions();

  const navigationRefs = {
    transformWrapperRef,
    preloadedDataRef,
    cachedEditStateRef,
    selectedImagePathRef,
    isBackendReadyRef,
    latestRenderedJobIdRef,
    previewJobIdRef,
    currentResRef,
    prevAdjustmentsRef,
  };

  const {
    handleGoHome,
    handleBackToLibrary,
    handleImageSelect,
    handleSelectSubfolder,
    handleSelectGooglePhotosAlbum,
    handleSelectAlbum,
    handleOpenFolder,
    handleContinueSession,
  } = useAppNavigation({
    clearThumbnailQueue,
    refs: navigationRefs,
  });

  useEffect(() => {
    if (!settingsPanelRequest) return;

    setUI({ activeView: 'library', isFullScreen: false });
    if (useEditorStore.getState().selectedImage) {
      handleBackToLibrary();
    }
  }, [settingsPanelRequest, handleBackToLibrary, setUI]);

  const {
    externalEditSession,
    isFinishing: isExternalEditFinishing,
    finishExternalEdit,
  } = useExternalEditSession(handleImageSelect);

  const {
    handleRate,
    handleClearSelection,
    handleLibraryImageSingleClick,
    handleImageClick,
    handleSetColorLabel,
    refreshAllFolderTrees,
    handleRemoveRecentFolder,
    handleCreateAlbumItem,
    handleRenameAlbumItem,
  } = useLibraryActions(handleImageSelect);

  const { displayList: sortedImageList, badges: groupBadgeInfo } = useSortedLibrary();

  const handleLibraryRefresh = useCallback(async () => {
    if (currentFolderPath === GOOGLE_PHOTOS_FOLDER_PATH) {
      await handleSelectGooglePhotosAlbum();
      return;
    }
    if (currentFolderPath) {
      if (currentFolderPath.startsWith('Album: ')) {
        const { activeAlbumId, albumTree } = useLibraryStore.getState();
        if (activeAlbumId) {
          const findObj = (nodes: any[]): any => {
            for (const n of nodes) {
              if (n.id === activeAlbumId) return n;
              if (n.type === 'group') {
                const f = findObj(n.children);
                if (f) return f;
              }
            }
            return null;
          };
          const album = findObj(albumTree);
          if (album) await handleSelectAlbum(album.id, album.name, album.images, true);
        }
      } else {
        await handleSelectSubfolder(currentFolderPath, false, undefined, false, true);
      }
    }
  }, [currentFolderPath, handleSelectSubfolder, handleSelectGooglePhotosAlbum, handleSelectAlbum]);

  const {
    executeDelete,
    handleDeleteSelected,
    handleCreateFolder,
    handleRenameFolder,
    handleSaveRename,
    handleRenameFiles,
    handleStartImport,
    handleImportClick,
    handlePasteFiles,
    handleMoveFilesToFolder,
  } = useFileOperations(
    handleLibraryRefresh,
    refreshAllFolderTrees,
    handleImageSelect,
    handleBackToLibrary,
    sortedImageList,
  );

  useEffect(() => {
    if (!rootPaths.length) return;

    const getElementUnderPointer = (event: PointerEvent) => document.elementFromPoint(event.clientX, event.clientY);
    const getDropTarget = (event: PointerEvent) => getFolderDropTargetPath(getElementUnderPointer(event));
    const isOverFolderSidebar = (event: PointerEvent) => isFolderSidebarTarget(getElementUnderPointer(event));

    const clearPointerDrag = () => {
      libraryImagePointerDragRef.current = null;
      setLibraryDragPreview(null);
      useLibraryStore.getState().setLibrary({ draggedImagePaths: [], dragTargetFolderPath: null });
    };

    const handleDocumentPointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return;

      const sourcePath = getLibraryImageDragSourcePath(event.target);
      if (!sourcePath) return;

      const { multiSelectedPaths } = useLibraryStore.getState();
      const draggedPaths = normalizeDraggedImagePaths(
        multiSelectedPaths.includes(sourcePath) ? multiSelectedPaths : [sourcePath],
      );

      if (draggedPaths.length === 0) return;

      libraryImagePointerDragRef.current = {
        dragging: false,
        paths: draggedPaths,
        pointerId: event.pointerId,
        sourcePath,
        startX: event.clientX,
        startY: event.clientY,
      };
    };

    const handleDocumentPointerMove = (event: PointerEvent) => {
      const drag = libraryImagePointerDragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;

      const distance = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
      if (!drag.dragging && distance < 4) return;

      if (!drag.dragging) {
        drag.dragging = true;
        useLibraryStore.getState().setLibrary({ draggedImagePaths: drag.paths, dragTargetFolderPath: null });
      }

      event.preventDefault();
      setLibraryDragPreview({ count: drag.paths.length, x: event.clientX, y: event.clientY });

      const destinationFolder = getDropTarget(event);
      const { dragTargetFolderPath, setLibrary } = useLibraryStore.getState();
      if (isOverFolderSidebar(event)) {
        if (dragTargetFolderPath !== destinationFolder) {
          setLibrary({ dragTargetFolderPath: destinationFolder });
        }
      } else if (dragTargetFolderPath !== null) {
        setLibrary({ dragTargetFolderPath: null });
      }
    };

    const handleDocumentPointerUp = (event: PointerEvent) => {
      const drag = libraryImagePointerDragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;

      const destinationFolder = drag.dragging && isOverFolderSidebar(event) ? getDropTarget(event) : null;
      const draggedPaths = drag.paths;
      const didDrag = drag.dragging;

      clearPointerDrag();

      if (!didDrag) return;

      event.preventDefault();
      event.stopPropagation();
      suppressNextClickRef.current = true;

      if (destinationFolder) {
        handleMoveFilesToFolder(draggedPaths, destinationFolder);
      }
    };

    const handleDocumentPointerCancel = (event: PointerEvent) => {
      const drag = libraryImagePointerDragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      clearPointerDrag();
    };

    const handleDocumentClick = (event: MouseEvent) => {
      if (!suppressNextClickRef.current) return;
      suppressNextClickRef.current = false;
      event.preventDefault();
      event.stopPropagation();
    };

    document.addEventListener('pointerdown', handleDocumentPointerDown, true);
    document.addEventListener('pointermove', handleDocumentPointerMove, { capture: true, passive: false });
    document.addEventListener('pointerup', handleDocumentPointerUp, true);
    document.addEventListener('pointercancel', handleDocumentPointerCancel, true);
    document.addEventListener('click', handleDocumentClick, true);

    return () => {
      document.removeEventListener('pointerdown', handleDocumentPointerDown, true);
      document.removeEventListener('pointermove', handleDocumentPointerMove, true);
      document.removeEventListener('pointerup', handleDocumentPointerUp, true);
      document.removeEventListener('pointercancel', handleDocumentPointerCancel, true);
      document.removeEventListener('click', handleDocumentClick, true);
    };
  }, [rootPaths.length, handleMoveFilesToFolder]);

  const {
    handleStartPanorama,
    handleSavePanorama,
    handleStartHdr,
    handleSaveHdr,
    handleApplyDenoise,
    handleBatchDenoise,
    handleSaveDenoisedImage,
    handleSaveCollage,
  } = useProductivityActions(handleLibraryRefresh);

  const {
    handleEditorContextMenu,
    handleThumbnailContextMenu,
    handleFolderTreeContextMenu,
    handleAlbumTreeContextMenu,
    handleMainLibraryContextMenu,
  } = useAppContextMenus({
    handleImageSelect,
    handleBackToLibrary,
    handleLibraryRefresh,
    handleRenameFiles,
    handleImportClick,
    refreshAllFolderTrees,
    refreshImageList: handleLibraryRefresh,
    executeDelete,
    handleRemoveRecentFolder,
  });

  useTauriListeners({
    refreshAllFolderTrees,
    handleSelectSubfolder,
    refreshImageList: handleLibraryRefresh,
    markGenerated,
  });

  useAndroidBackHandler();

  const handleToggleFullScreen = useCallback(() => {
    const { zoom, selectedImage } = useEditorStore.getState();
    const currentlyZoomed = zoom > 1.01;
    setUI({ isInstantTransition: currentlyZoomed });

    if (isFullScreen) {
      setUI({ isFullScreen: false });
    } else {
      if (!selectedImage) return;
      setUI({ isFullScreen: true });
    }

    if (currentlyZoomed) {
      setTimeout(() => setUI({ isInstantTransition: false }), 100);
    }
  }, [isFullScreen, setUI]);

  useKeyboardShortcuts({
    sortedImageList,
    handleBackToLibrary,
    handleDeleteSelected,
    handleImageSelect,
    handlePasteFiles,
    handleToggleFullScreen,
    handleZoomChange,
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateViewportSize = () => {
      const nextViewportSize = {
        width: Math.round(window.visualViewport?.width ?? window.innerWidth),
        height: Math.round(window.visualViewport?.height ?? window.innerHeight),
      };

      setViewportSize((prev) =>
        prev.width === nextViewportSize.width && prev.height === nextViewportSize.height ? prev : nextViewportSize,
      );
    };

    updateViewportSize();

    window.addEventListener('resize', updateViewportSize);
    window.addEventListener('orientationchange', updateViewportSize);
    window.visualViewport?.addEventListener('resize', updateViewportSize);

    return () => {
      window.removeEventListener('resize', updateViewportSize);
      window.removeEventListener('orientationchange', updateViewportSize);
      window.visualViewport?.removeEventListener('resize', updateViewportSize);
    };
  }, []);

  useEffect(() => {
    const handleGlobalContextMenu = (event: MouseEvent) => {
      event.preventDefault();
    };
    window.addEventListener('contextmenu', handleGlobalContextMenu);
    return () => window.removeEventListener('contextmenu', handleGlobalContextMenu);
  }, []);

  const isLightTheme = useMemo(() => [Theme.Light, Theme.Snow, Theme.Arctic].includes(theme as Theme), [theme]);

  useEffect(() => {
    if (
      (activeRightPanel !== Panel.Masks || !activeMaskContainerId) &&
      (activeRightPanel !== Panel.Ai || !activeAiPatchContainerId)
    ) {
      setEditor({ isMaskControlHovered: false });
    }
  }, [activeRightPanel, activeMaskContainerId, activeAiPatchContainerId, setEditor]);

  useEffect(() => {
    const unlisten = listen('ai-connector-status-update', (event: any) => {
      setEditor({ isAIConnectorConnected: event.payload.connected });
    });
    invoke(Invokes.CheckAIConnectorStatus);
    const interval = setInterval(() => invoke(Invokes.CheckAIConnectorStatus), 10000);
    return () => {
      clearInterval(interval);
      unlisten.then((f) => f());
    };
  }, [setEditor]);

  const createResizeHandler = (stateKey: string, startSize: number) => (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);

    const pointerId = e.pointerId;
    const target = e.currentTarget;
    const startX = e.clientX;
    const startY = e.clientY;

    const previousTouchAction = document.documentElement.style.touchAction;
    const previousUserSelect = document.documentElement.style.userSelect;

    target.setPointerCapture?.(pointerId);
    document.documentElement.style.touchAction = 'none';
    document.documentElement.style.userSelect = 'none';

    const doDrag = (moveEvent: PointerEvent) => {
      if (moveEvent.pointerId !== pointerId) return;
      moveEvent.preventDefault();

      if (stateKey === 'left') {
        let w = startSize + (moveEvent.clientX - startX);
        if (w < 200) w = 48;
        else if (w > 600) w = 600;
        setUI({ leftPanelWidth: Math.round(w) });
      } else if (stateKey === 'right') {
        let w = startSize - (moveEvent.clientX - startX);
        if (w < 200) w = 48;
        else if (w > 600) w = 600;
        setUI({ rightPanelWidth: Math.round(w) });
      } else if (stateKey === 'bottom') {
        const newHeight = startSize - (moveEvent.clientY - startY);
        if (newHeight < 100) {
          setUI((state) => ({
            uiVisibility: { ...state.uiVisibility, filmstrip: false },
          }));
        } else {
          setUI((state) => ({
            bottomPanelHeight: Math.round(Math.min(newHeight, 400)),
            uiVisibility: { ...state.uiVisibility, filmstrip: true },
          }));
        }
      } else if (stateKey === 'compact') {
        setUI({
          compactEditorPanelHeightOverride: Math.round(
            Math.max(
              compactEditorPanelMinHeight,
              Math.min(startSize - (moveEvent.clientY - startY), compactEditorPanelMaxHeight),
            ),
          ),
        });
      }
    };

    const stopDrag = (upEvent: PointerEvent) => {
      if (upEvent.pointerId !== pointerId) return;
      if (target.hasPointerCapture?.(pointerId)) target.releasePointerCapture(pointerId);

      document.documentElement.style.cursor = '';
      document.documentElement.style.touchAction = previousTouchAction;
      document.documentElement.style.userSelect = previousUserSelect;

      window.removeEventListener('pointermove', doDrag);
      window.removeEventListener('pointerup', stopDrag);
      window.removeEventListener('pointercancel', stopDrag);
      setIsResizing(false);
    };
    document.documentElement.style.cursor =
      stateKey === 'bottom' || stateKey === 'compact' ? 'row-resize' : 'col-resize';

    window.addEventListener('pointermove', doDrag, { passive: false });
    window.addEventListener('pointerup', stopDrag);
    window.addEventListener('pointercancel', stopDrag);
  };

  useEffect(() => {
    const appWindow = getCurrentWindow();
    const checkFullscreen = async () => {
      setUI({ isWindowFullScreen: await appWindow.isFullscreen() });
    };
    checkFullscreen();
    const unlistenPromise = appWindow.onResized(checkFullscreen);
    return () => {
      unlistenPromise.then((unlisten: any) => unlisten());
    };
  }, [setUI]);

  const handleRightPanelSelect = useCallback(
    (panelId: Panel) => {
      setRightPanel(panelId);
      setEditor({ activeMaskId: null, activeAiSubMaskId: null, isWbPickerActive: false });
    },
    [setRightPanel, setEditor],
  );

  const handleToggleFolder = useCallback(
    async (path: string) => {
      const isExpanding = !expandedFolders.has(path);
      setLibrary((state) => {
        const newSet = new Set(state.expandedFolders);
        if (isExpanding) {
          newSet.add(path);
        } else {
          newSet.delete(path);
        }
        return { expandedFolders: newSet };
      });
      if (!isExpanding) return;
      try {
        const showCounts = appSettings?.enableFolderImageCounts ?? false;
        const newChildren: any[] = await invoke(Invokes.GetFolderChildren, {
          path,
          showImageCounts: showCounts,
        });
        setLibrary((state) => ({
          folderTrees: state.folderTrees.map((t: any) => insertChildrenIntoTree(t, path, newChildren)),
        }));
      } catch (err) {
        toast.error(`Failed to load folder: ${err}`);
      }
    },
    [expandedFolders, appSettings?.enableFolderImageCounts, setLibrary],
  );

  const renderAppPanel = useCallback(
    (panelId: Panel) => {
      switch (panelId) {
        case Panel.FolderTree:
          return (
            <FolderTree
              isResizing={isResizing}
              onContextMenu={handleFolderTreeContextMenu}
              onAlbumContextMenu={handleAlbumTreeContextMenu}
              onSelectAlbum={handleSelectAlbum}
              onFolderSelect={(path, options) => handleSelectSubfolder(path, !!options?.asSessionRoot)}
              onGooglePhotosSelect={handleSelectGooglePhotosAlbum}
              onRecentFolderRemove={handleRemoveRecentFolder}
              onToggleFolder={handleToggleFolder}
              onOpenFolder={handleOpenFolder}
              style={{ width: '100%', height: '100%' }}
              isInstantTransition={isInstantTransition}
            />
          );
        case Panel.Export:
          return (
            <ExportPanel
              exportState={exportState}
              multiSelectedPaths={multiSelectedPaths}
              selectedImage={selectedImage}
              setExportState={setExportState}
              appSettings={appSettings}
              onSettingsChange={handleSettingsChange}
              rootPaths={rootPaths}
              isVisible={true}
              onClose={() => setUI({ isLibraryExportPanelVisible: false })}
            />
          );
        case Panel.Adjustments:
          return <Controls />;
        case Panel.Metadata:
          return <MetadataPanel />;
        case Panel.Crop:
          return <CropPanel />;
        case Panel.Masks:
          return <MasksPanel />;
        case Panel.Ai:
          return <AIPanel />;
        case Panel.Presets:
          return <PresetsPanel onNavigateToCommunity={() => setUI({ activeView: 'community' })} />;
        case Panel.History:
          return <HistoryPanel />;
        default:
          return null;
      }
    },
    [
      isResizing,
      handleFolderTreeContextMenu,
      handleAlbumTreeContextMenu,
      handleSelectAlbum,
      handleSelectSubfolder,
      handleSelectGooglePhotosAlbum,
      handleRemoveRecentFolder,
      handleToggleFolder,
      handleOpenFolder,
      setUI,
      isInstantTransition,
      exportState,
      multiSelectedPaths,
      selectedImage,
      setExportState,
      appSettings,
      handleSettingsChange,
      rootPaths,
    ],
  );

  const hasRoots = rootPaths && rootPaths.length > 0;
  const handleOpenListedFolder = useCallback(
    async (path: string) => {
      const { rootPaths: currentRoots, folderTrees, setLibrary } = useLibraryStore.getState();
      if (!currentRoots.includes(path)) {
        const nextRoots = [...currentRoots, path];
        setLibrary({ rootPaths: nextRoots, isTreeLoading: true });
        if (appSettings) {
          await handleSettingsChange({ ...appSettings, rootFolders: nextRoots } as any);
        }
        try {
          const newTree = await invoke(Invokes.GetFolderTree, {
            path,
            expandedFolders: [path],
            showImageCounts: appSettings?.enableFolderImageCounts ?? false,
          });
          setLibrary({ folderTrees: [...folderTrees, newTree] });
        } catch (err) {
          toast.error(`Failed to load folder tree: ${err}`);
        } finally {
          setLibrary({ isTreeLoading: false });
        }
      }
      await handleSelectSubfolder(path, true);
    },
    [appSettings, handleSettingsChange, handleSelectSubfolder],
  );
  const hasMainContent = hasRoots || (activeView === 'editor' && !!selectedImage);

  const shouldHideFolderTree = isAndroid;
  const isWgpuActive =
    activeView === 'editor' &&
    appSettings?.useWgpuRenderer !== false &&
    selectedImage?.isReady &&
    hasRenderedFirstFrame;
  const useMacWindowShell = osPlatform === 'macos' && !appSettings?.decorations && !isWindowFullScreen && !isFullScreen;

  const layoutSensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));
  const handleDragStart = (e: any) => {
    if (e.active.data.current?.type === 'layout-tab') {
      setLayoutDragItem(e.active.data.current.panel as Panel);
    }
  };
  const handleDragEnd = (e: any) => {
    setLayoutDragItem(null);
    if (e.active.data.current?.type === 'layout-tab' && e.over?.data.current?.type === 'layout-region') {
      movePanel(e.active.data.current.panel as Panel, e.over.data.current.region as PanelRegion);
    }
  };
  const ActiveOverlayIcon = activeLayoutDragItem ? PANEL_ICONS[activeLayoutDragItem] : null;

  return (
    <>
      <ImageProcessingManager
        transformWrapperRef={transformWrapperRef}
        prevAdjustmentsRef={prevAdjustmentsRef}
        previewJobIdRef={previewJobIdRef}
        latestRenderedJobIdRef={latestRenderedJobIdRef}
        currentResRef={currentResRef}
      />
      <ImageLoaderManager cachedEditStateRef={cachedEditStateRef} />
      <div
        className={clsx(
          'flex flex-col h-screen font-sans text-text-primary overflow-hidden select-none',
          useMacWindowShell && 'macos-window-shell',
          isWgpuActive ? 'bg-transparent' : 'bg-bg-primary',
        )}
      >
        <div
          className={clsx(
            'shrink-0 overflow-hidden z-50',
            !isInstantTransition && 'transition-all duration-300 ease-in-out',
            isFullScreen ? 'max-h-0 opacity-0 pointer-events-none' : 'max-h-[60px] opacity-100',
          )}
        >
          {appSettings?.decorations || (!isWindowFullScreen && <TitleBar />)}
        </div>
        <div
          className={clsx(
            'flex-1 flex flex-col min-h-0',
            isLayoutReady && hasMainContent && !isInstantTransition && 'transition-all duration-300 ease-in-out',
            [hasMainContent && (isFullScreen ? 'p-0 gap-0' : 'p-2 gap-2')],
          )}
        >
          <DndContext sensors={layoutSensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
            <div className="flex flex-row grow h-full min-h-0">
              {!shouldHideFolderTree && hasMainContent && (
                <SidePanelArea
                  side="left"
                  width={leftPanelWidth}
                  topRegion="leftTop"
                  bottomRegion="leftBottom"
                  renderPanel={renderAppPanel}
                  onWidthChange={createResizeHandler('left', leftPanelWidth)}
                  isResizing={isResizing}
                />
              )}
              <div className="relative flex-1 flex flex-col min-w-0">
                {selectedImage && externalEditSession && (
                  <ExternalEditBar
                    session={externalEditSession}
                    isFinishing={isExternalEditFinishing}
                    errorMessage={exportState.status === Status.Error ? exportState.errorMessage : ''}
                    onDone={finishExternalEdit}
                  />
                )}
                <div
                  className={clsx(
                    'flex-1 flex flex-col min-w-0 h-full',
                    activeView === 'editor' && selectedImage ? 'flex' : 'hidden',
                  )}
                >
                  {selectedImage && (
                    <EditorView
                      transformWrapperRef={transformWrapperRef}
                      isResizing={isResizing}
                      isCompactPortrait={isCompactPortrait}
                      isAndroid={isAndroid}
                      compactEditorPanelHeight={compactEditorPanelHeight}
                      compactEditorPanelCollapsedHeight={compactEditorPanelCollapsedHeight}
                      thumbnailAspectRatio={thumbnailAspectRatio}
                      sortedImageList={sortedImageList}
                      createResizeHandler={createResizeHandler}
                      handleBackToLibrary={handleBackToLibrary}
                      handleEditorContextMenu={handleEditorContextMenu}
                      handleThumbnailContextMenu={handleThumbnailContextMenu}
                      handleMainLibraryContextMenu={handleMainLibraryContextMenu}
                      handleImageClick={handleImageClick}
                      handleClearSelection={handleClearSelection}
                      handleCopyAdjustments={handleCopyAdjustments}
                      handlePasteAdjustments={handlePasteAdjustments}
                      handleRate={handleRate}
                      handleZoomChange={handleZoomChange}
                      handleRightPanelSelect={handleRightPanelSelect}
                      requestThumbnails={requestThumbnails}
                    />
                  )}
                </div>
                <div
                  className={clsx(
                    'flex-1 flex flex-col min-w-0 h-full',
                    activeView === 'editor' && selectedImage ? 'hidden' : 'flex',
                  )}
                >
                  <LibraryView
                    sortedImageList={sortedImageList}
                    groupBadgeInfo={groupBadgeInfo}
                    thumbnailSize={thumbnailSize}
                    thumbnailAspectRatio={thumbnailAspectRatio}
                    libraryViewMode={libraryViewMode}
                    isAndroid={isAndroid}
                    setThumbnailSize={setThumbnailSize}
                    setThumbnailAspectRatio={setThumbnailAspectRatio}
                    setLibraryViewMode={setLibraryViewMode}
                    handleClearSelection={handleClearSelection}
                    handleLibraryImageSingleClick={handleLibraryImageSingleClick}
                    handleImageSelect={handleImageSelect}
                    handleRate={handleRate}
                    handleThumbnailContextMenu={handleThumbnailContextMenu}
                    handleMainLibraryContextMenu={handleMainLibraryContextMenu}
                    handleContinueSession={handleContinueSession}
                     handleGoHome={handleGoHome}
                     handleOpenFolder={handleOpenFolder}
                     handleOpenListedFolder={handleOpenListedFolder}
                     handleImportClick={handleImportClick}
                    handleLibraryRefresh={handleLibraryRefresh}
                    handleCopyAdjustments={handleCopyAdjustments}
                    handlePasteAdjustments={handlePasteAdjustments}
                    handleResetAdjustments={handleResetAdjustments}
                    requestThumbnails={requestThumbnails}
                  />
                </div>
                {isSettingsOpen && appSettings && hasRoots && (
                  <div className="absolute inset-0 z-50 flex bg-bg-secondary rounded-lg">
                    <div className="w-full h-full flex flex-col p-4 lg:p-8 overflow-y-auto custom-scrollbar">
                      <SettingsPanel
                        appSettings={appSettings}
                        initialCategory={settingsPanelRequest?.category}
                        initialCategoryRequestId={settingsPanelRequest?.id}
                        onBack={() => setUI({ isSettingsOpen: false })}
                        onLibraryRefresh={handleLibraryRefresh}
                        onSettingsChange={handleSettingsChange}
                        rootPaths={rootPaths}
                      />
                    </div>
                  </div>
                )}
              </div>
              {!isAndroid && hasMainContent && (
                <SidePanelArea
                  side="right"
                  width={rightPanelWidth}
                  topRegion="rightTop"
                  bottomRegion="rightBottom"
                  renderPanel={renderAppPanel}
                  onWidthChange={createResizeHandler('right', rightPanelWidth)}
                  isResizing={isResizing}
                />
              )}
            </div>
            <DragOverlay dropAnimation={{ duration: 150, easing: 'cubic-bezier(0.18, 0.67, 0.6, 1.22)' }}>
              {activeLayoutDragItem && ActiveOverlayIcon ? (
                <div className="w-10 h-10 bg-surface shadow-2xl rounded-md flex items-center justify-center text-text-primary ring-1 ring-border-color">
                  <ActiveOverlayIcon size={20} />
                </div>
              ) : null}
            </DragOverlay>
          </DndContext>
        </div>
        <AppModals
          handleImageSelect={handleImageSelect}
          handleSavePanorama={handleSavePanorama}
          handleStartPanorama={handleStartPanorama}
          handleSaveHdr={handleSaveHdr}
          handleStartHdr={handleStartHdr}
          refreshImageList={handleLibraryRefresh}
          handleApplyDenoise={handleApplyDenoise}
          handleBatchDenoise={handleBatchDenoise}
          handleSaveDenoisedImage={handleSaveDenoisedImage}
          handleCreateFolder={handleCreateFolder}
          handleRenameFolder={handleRenameFolder}
          handleSaveRename={handleSaveRename}
          handleStartImport={handleStartImport}
          handleSetColorLabel={handleSetColorLabel}
          handleRate={handleRate}
          executeDelete={executeDelete}
          handleSaveCollage={handleSaveCollage}
          handleCreateAlbumItem={handleCreateAlbumItem}
          handleRenameAlbumItem={handleRenameAlbumItem}
        />
        {libraryDragPreview && (
          <div
            className="fixed z-9999 pointer-events-none rounded-md bg-surface/95 border border-accent px-3 py-2 shadow-2xl text-sm font-semibold text-text-primary"
            style={{
              left: libraryDragPreview.x + 14,
              top: libraryDragPreview.y + 14,
            }}
          >
            {libraryDragPreview.count === 1 ? 'Moving 1 image' : `Moving ${libraryDragPreview.count} images`}
          </div>
        )}
        <ToastContainer
          position="bottom-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable={false}
          pauseOnHover
          theme={isLightTheme ? 'light' : 'dark'}
          transition={Slide}
          toastClassName={() =>
            clsx(
              'relative flex min-h-16 p-4 rounded-lg justify-between overflow-hidden cursor-pointer mb-4',
              'bg-surface! text-text-primary! border! border-border-color! shadow-2xl! max-w-[420px]!',
            )
          }
        />
      </div>
    </>
  );
}

const AppWrapper = () => (
  <ClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY} routerPush={(to) => {}} routerReplace={(to) => {}}>
    <ContextMenuProvider>
      <App />
      <GlobalTooltip />
    </ContextMenuProvider>
  </ClerkProvider>
);

export default AppWrapper;
