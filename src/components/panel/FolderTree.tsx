import {
  Cloud,
  Folder,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Search,
  X,
  History,
} from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useMemo, useEffect, useRef } from 'react';
import Text from '../ui/Text';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../types/typography';

// Import our stores
import { useLibraryStore } from '../../store/useLibraryStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { GOOGLE_PHOTOS_FOLDER_PATH } from '../../store/useGooglePhotosStore';
import { getFolderDisplayName, isSameFolderPath, normalizeFolderPath } from '../../utils/folderPaths';

export interface FolderTree {
  children: FolderTree[];
  isDir: boolean; // Fixed camelCase
  name: string;
  path: string;
  imageCount?: number;
  hasSubdirs?: boolean; // Fixed camelCase
}

// Cleaned up props! We only pass functions and UI layout state
interface FolderTreeProps {
  isResizing: boolean;
  isVisible: boolean;
  onContextMenu(event: any, path: string | null, isPinned?: boolean, isRecent?: boolean): void;
  onFolderSelect(folder: string, options?: { asSessionRoot?: boolean }): void;
  onGooglePhotosSelect(): void;
  onRecentFolderRemove(folder: string): void;
  onToggleFolder(folder: string): void;
  setIsVisible(visible: boolean): void;
  style: any;
  isInstantTransition: boolean;
}

interface TreeNodeProps {
  expandedFolders: Set<string>;
  isExpanded: boolean;
  node: FolderTree;
  onContextMenu(event: any, path: string, isPinned?: boolean): void;
  onFolderSelect(folder: string, options?: { asSessionRoot?: boolean }): void;
  onToggle(path: string): void;
  selectedPath: string | null;
  pinnedFolders: string[];
  selectAsSessionRoot: boolean;
  showImageCounts: boolean;
  isInstantTransition: boolean;
  dragTargetFolderPath: string | null;
  hasActiveImageDrag: boolean;
}

interface VisibleProps {
  index: number;
  total: number;
}

const filterTree = (node: FolderTree | null, query: string): FolderTree | null => {
  if (!node) {
    return null;
  }

  const lowerCaseQuery = query.toLowerCase();
  const isMatch = node.name.toLowerCase().includes(lowerCaseQuery);

  if (!node.children || node.children.length === 0) {
    return isMatch ? node : null;
  }

  const filteredChildren = node.children
    .map((child: FolderTree) => filterTree(child, query))
    .filter((child: FolderTree | null): child is FolderTree => child !== null);

  if (isMatch || filteredChildren.length > 0) {
    return { ...node, children: filteredChildren };
  }

  return null;
};

const getAutoExpandedPaths = (node: FolderTree, paths: Set<string>) => {
  if (node.children && node.children.length > 0) {
    paths.add(node.path);
    node.children.forEach((child: FolderTree) => getAutoExpandedPaths(child, paths));
  }
};

function RecentFolderRow({
  path,
  isSelected,
  onContextMenu,
  onFolderSelect,
  onRemove,
}: {
  path: string;
  isSelected: boolean;
  onContextMenu(event: any, path: string, isPinned?: boolean, isRecent?: boolean): void;
  onFolderSelect(folder: string, options?: { asSessionRoot?: boolean }): void;
  onRemove(folder: string): void;
}) {
  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        className={clsx('group flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected,
          'hover:bg-card-active': !isSelected,
        })}
        onClick={() => onFolderSelect(path, { asSessionRoot: true })}
        onContextMenu={(e: any) => onContextMenu(e, path, false, true)}
        data-folder-path={path}
      >
        <div className="p-0.5 rounded-sm transition-colors text-text-secondary">
          <History size={16} className="shrink-0" />
        </div>
        <span className="truncate select-none flex-1" title={path}>
          {getFolderDisplayName(path)}
        </span>
        <button
          type="button"
          className="p-0.5 rounded-sm text-text-secondary opacity-0 group-hover:opacity-100 hover:bg-surface-hover transition"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(path);
          }}
          data-tooltip="Remove from recent"
        >
          <X size={14} />
        </button>
      </div>
    </Text>
  );
}

function TreeNode({
  expandedFolders,
  isExpanded,
  node,
  onContextMenu,
  onFolderSelect,
  onToggle,
  selectedPath,
  pinnedFolders,
  selectAsSessionRoot,
  showImageCounts,
  isInstantTransition,
  dragTargetFolderPath,
  hasActiveImageDrag,
}: TreeNodeProps) {
  const hasChildren = node.hasSubdirs || (node.children && node.children.length > 0);
  const isSelected = node.path === selectedPath;
  const isPinned = pinnedFolders.includes(node.path);
  const isDropTarget = dragTargetFolderPath === node.path;
  const expandOnHoverTimer = useRef<number | null>(null);

  const clearExpandOnHoverTimer = () => {
    if (expandOnHoverTimer.current !== null) {
      window.clearTimeout(expandOnHoverTimer.current);
      expandOnHoverTimer.current = null;
    }
  };

  useEffect(() => clearExpandOnHoverTimer, []);

  useEffect(() => {
    const shouldExpandOnHover = hasActiveImageDrag && dragTargetFolderPath === node.path && hasChildren && !isExpanded;
    if (!shouldExpandOnHover) {
      clearExpandOnHoverTimer();
      return;
    }

    if (expandOnHoverTimer.current === null) {
      expandOnHoverTimer.current = window.setTimeout(() => {
        onToggle(node.path);
        expandOnHoverTimer.current = null;
      }, 700);
    }

    return clearExpandOnHoverTimer;
  }, [dragTargetFolderPath, hasActiveImageDrag, hasChildren, isExpanded, node.path, onToggle]);

  const handleFolderIconClick = (e: any) => {
    e.stopPropagation();
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const handleNameClick = () => {
    onFolderSelect(node.path, { asSessionRoot: selectAsSessionRoot });
  };

  const handleNameDoubleClick = () => {
    if (hasChildren) {
      onToggle(node.path);
    }
  };

  const containerVariants: any = {
    closed: { height: 0, opacity: 0, transition: { duration: 0.2, ease: 'easeInOut' } },
    open: { height: 'auto', opacity: 1, transition: { duration: 0.25, ease: 'easeInOut' } },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -15 },
    visible: ({ index, total }: VisibleProps) => ({
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.25,
        delay: total < 8 ? index * 0.05 : 0,
      },
    }),
    exit: { opacity: 0, x: -15, transition: { duration: 0.2 } },
  };

  return (
    <Text as="div" color={TextColors.primary} weight={TextWeights.medium}>
      <div
        className={clsx('flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer', {
          'bg-surface': isSelected,
          'ring-2 ring-inset ring-accent bg-accent/20 shadow-lg': isDropTarget,
          'hover:bg-card-active': !isSelected,
        })}
        onClick={handleNameClick}
        onContextMenu={(e: any) => onContextMenu(e, node.path, isPinned)}
        data-folder-path={node.path}
      >
        <div
          className={clsx('p-0.5 rounded-sm transition-colors', {
            [TEXT_COLOR_KEYS[TextColors.secondary]]: !isExpanded,
            'hover:bg-surface-hover': !isSelected && hasChildren,
          })}
          onClick={handleFolderIconClick}
        >
          {isExpanded ? <FolderOpen size={16} /> : <Folder size={16} />}
        </div>

        <span onDoubleClick={handleNameDoubleClick} className="truncate select-none flex-1">
          <span className="truncate">{node.name}</span>
          {typeof node.imageCount === 'number' && node.imageCount > 0 && (
            <Text
              as="span"
              variant={TextVariants.small}
              color={TextColors.secondary}
              className={clsx(
                'inline-block ml-1 transition-all ease-in-out duration-300',
                showImageCounts ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-2',
              )}
            >
              ({node.imageCount})
            </Text>
          )}
        </span>

        {hasChildren && (
          <Text
            as="div"
            color={TextColors.secondary}
            className="p-0.5 rounded-sm hover:bg-surface/50"
            onClick={handleFolderIconClick}
          >
            {isExpanded ? <ChevronUp size={16} className="shrink-0" /> : <ChevronDown size={16} className="shrink-0" />}
          </Text>
        )}
      </div>

      <AnimatePresence initial={false}>
        {hasChildren && isExpanded && node.children && node.children.length > 0 && (
          <motion.div
            animate="open"
            className="pl-4 border-l-[1.5px] border-border-color/50 ml-3.75 overflow-hidden"
            exit="closed"
            initial={isInstantTransition ? 'open' : 'closed'}
            key="children-container"
            variants={containerVariants}
          >
            <div className="py-1">
              <AnimatePresence>
                {node?.children?.map((childNode: any, index: number) => (
                  <motion.div
                    animate="visible"
                    custom={{ index, total: node.children.length }}
                    exit="exit"
                    initial={isInstantTransition ? 'visible' : 'hidden'}
                    key={childNode.path}
                    layout={isInstantTransition ? false : 'position'}
                    variants={itemVariants}
                  >
                    <TreeNode
                      expandedFolders={expandedFolders}
                      isExpanded={expandedFolders.has(childNode.path)}
                      node={childNode}
                      onContextMenu={onContextMenu}
                      onFolderSelect={onFolderSelect}
                      onToggle={onToggle}
                      selectedPath={selectedPath}
                      pinnedFolders={pinnedFolders}
                      selectAsSessionRoot={selectAsSessionRoot}
                      showImageCounts={showImageCounts}
                      isInstantTransition={isInstantTransition}
                      dragTargetFolderPath={dragTargetFolderPath}
                      hasActiveImageDrag={hasActiveImageDrag}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Text>
  );
}

export default function FolderTree({
  isResizing,
  isVisible,
  onContextMenu,
  onFolderSelect,
  onGooglePhotosSelect,
  onRecentFolderRemove,
  onToggleFolder,
  setIsVisible,
  style,
  isInstantTransition,
}: FolderTreeProps) {
  // Grab state directly from stores
  const { appSettings } = useSettingsStore();
  const {
    folderTree: tree,
    pinnedFolderTrees,
    currentFolderPath: selectedPath,
    rootPath,
    expandedFolders,
    isTreeLoading: isLoading,
    draggedImagePaths,
    dragTargetFolderPath,
  } = useLibraryStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [isHovering, setIsHovering] = useState(false);

  // Derive variables from settings
  const pinnedFolders = appSettings?.pinnedFolders || [];
  const showImageCounts = appSettings?.enableFolderImageCounts ?? false;
  const showRecentFolders = appSettings?.showRecentFolders ?? true;
  const googlePhotosEnabled = appSettings?.googlePhotosIntegrationEnabled ?? false;
  const googlePhotosAlbumTitle = appSettings?.googlePhotosAlbumTitle || 'RapidRaw';

  const handleEmptyAreaContextMenu = (e: any) => {
    if (e.target === e.currentTarget) {
      onContextMenu(e, null, false);
    }
  };

  const trimmedQuery = searchQuery.trim();
  const isSearching = trimmedQuery.length > 1;

  const filteredTree = useMemo(() => {
    if (!isSearching) return tree;
    return filterTree(tree, trimmedQuery);
  }, [tree, trimmedQuery, isSearching]);

  const filteredPinnedTrees = useMemo(() => {
    if (!isSearching) return pinnedFolderTrees;
    return pinnedFolderTrees
      .map((pinnedTree) => filterTree(pinnedTree, trimmedQuery))
      .filter((t): t is FolderTree => t !== null);
  }, [pinnedFolderTrees, trimmedQuery, isSearching]);

  const visibleRecentFolders = useMemo(() => {
    if (!showRecentFolders) return [];
    const pinnedSet = new Set(pinnedFolders.map(normalizeFolderPath));
    const seen = new Set<string>();
    return (appSettings?.recentFolders || []).filter((path: string) => {
      const normalized = normalizeFolderPath(path);
      if (!path || seen.has(normalized) || pinnedSet.has(normalized)) {
        return false;
      }
      seen.add(normalized);
      return !isSearching || getFolderDisplayName(path).toLowerCase().includes(trimmedQuery.toLowerCase()) || path.toLowerCase().includes(trimmedQuery.toLowerCase());
    });
  }, [appSettings?.recentFolders, isSearching, pinnedFolders, rootPath, showRecentFolders, trimmedQuery]);

  const searchAutoExpandedFolders = useMemo(() => {
    if (!isSearching) {
      return new Set<string>();
    }
    const newExpanded = new Set<string>();
    if (filteredTree) {
      getAutoExpandedPaths(filteredTree, newExpanded);
    }
    filteredPinnedTrees.forEach((pinned) => {
      getAutoExpandedPaths(pinned, newExpanded);
    });
    return newExpanded;
  }, [isSearching, filteredTree, filteredPinnedTrees]);

  const effectiveExpandedFolders = useMemo(() => {
    return new Set([...expandedFolders, ...searchAutoExpandedFolders]);
  }, [expandedFolders, searchAutoExpandedFolders]);

  const hasVisiblePinnedTrees = filteredPinnedTrees && filteredPinnedTrees.length > 0;
  const hasVisibleRecentFolders = visibleRecentFolders.length > 0;
  const hasActiveImageDrag = draggedImagePaths.length > 0;

  return (
    <div
      className={clsx(
        'relative bg-bg-secondary rounded-lg shrink-0',
        !isResizing && 'transition-[width] duration-300 ease-in-out',
      )}
      style={style}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      data-folder-sidebar
    >
      {!isVisible && (
        <button
          className="absolute top-1/2 -translate-y-1/2 right-1 w-6 h-10 hover:bg-card-active rounded-md flex items-center justify-center z-30"
          onClick={() => setIsVisible(true)}
          data-tooltip="Expand"
        >
          <ChevronRight size={16} />
        </button>
      )}

      {isVisible && (
        <div className="p-2 flex flex-col h-full">
          <div className="pt-1 pb-2">
            <div className="flex items-center">
              <AnimatePresence>
                {isHovering && (
                  <motion.button
                    initial={{ width: 0, padding: 0, marginRight: 0, opacity: 0 }}
                    animate={{ width: 36, padding: 10, marginRight: 6, opacity: 1 }}
                    exit={{ width: 0, padding: 0, marginRight: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeInOut' }}
                    className="bg-surface rounded-md hover:bg-card-active flex items-center justify-center shrink-0 overflow-hidden transition-colors"
                    onClick={() => setIsVisible(false)}
                    data-tooltip="Collapse"
                  >
                    <ChevronLeft size={17.5} className="text-text-secondary shrink-0" />
                  </motion.button>
                )}
              </AnimatePresence>
              <div className="relative flex-1 min-w-0">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
                <input
                  type="text"
                  placeholder="Search folders..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-surface border border-transparent rounded-md pl-9 pr-8 py-2 text-sm focus:outline-hidden"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-card-active"
                    data-tooltip="Clear search"
                  >
                    <X size={16} className="text-text-secondary" />
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto" onContextMenu={handleEmptyAreaContextMenu}>
            {filteredTree && (
              <div className="pb-2">
                <Text
                  as="div"
                  variant={TextVariants.small}
                  weight={TextWeights.bold}
                  className="w-full px-2 py-1.5 uppercase tracking-wider select-none text-text-secondary"
                >
                  Current Folder
                </Text>
                <TreeNode
                  expandedFolders={effectiveExpandedFolders}
                  isExpanded={effectiveExpandedFolders.has(filteredTree.path)}
                  node={filteredTree}
                  onContextMenu={onContextMenu}
                  onFolderSelect={onFolderSelect}
                  onToggle={onToggleFolder}
                  selectedPath={selectedPath}
                  pinnedFolders={pinnedFolders}
                  selectAsSessionRoot={false}
                  showImageCounts={showImageCounts && isHovering}
                  isInstantTransition={isInstantTransition}
                  dragTargetFolderPath={dragTargetFolderPath}
                  hasActiveImageDrag={hasActiveImageDrag}
                />
              </div>
            )}

            {(hasVisiblePinnedTrees || hasVisibleRecentFolders) && filteredTree && (
              <div className="h-px bg-border-color my-1 mx-2" />
            )}

            <div className="pb-2 space-y-0.5">
              {(hasVisiblePinnedTrees || hasVisibleRecentFolders) && (
                <Text
                  as="div"
                  variant={TextVariants.small}
                  weight={TextWeights.bold}
                  className="w-full px-2 py-1.5 uppercase tracking-wider select-none text-text-secondary"
                >
                  Quick Access
                </Text>
              )}
              {hasVisiblePinnedTrees &&
                filteredPinnedTrees.map((pinnedTree) => (
                  <TreeNode
                    key={pinnedTree.path}
                    expandedFolders={effectiveExpandedFolders}
                    isExpanded={effectiveExpandedFolders.has(pinnedTree.path)}
                    node={pinnedTree}
                    onContextMenu={onContextMenu}
                    onFolderSelect={onFolderSelect}
                    onToggle={onToggleFolder}
                    selectedPath={selectedPath}
                    pinnedFolders={pinnedFolders}
                    selectAsSessionRoot={true}
                    showImageCounts={showImageCounts && isHovering}
                    isInstantTransition={isInstantTransition}
                    dragTargetFolderPath={dragTargetFolderPath}
                    hasActiveImageDrag={hasActiveImageDrag}
                  />
                ))}

              {hasVisibleRecentFolders &&
                visibleRecentFolders.map((recentFolder) => (
                  <RecentFolderRow
                    key={recentFolder}
                    path={recentFolder}
                    isSelected={isSameFolderPath(recentFolder, selectedPath)}
                    onContextMenu={onContextMenu}
                    onFolderSelect={onFolderSelect}
                    onRemove={onRecentFolderRemove}
                  />
                ))}
            </div>

            {!filteredTree && !hasVisiblePinnedTrees && !hasVisibleRecentFolders && isSearching && (
              <Text className="p-2 text-center">No folders found.</Text>
            )}

            {!tree && pinnedFolderTrees.length === 0 && !hasVisibleRecentFolders && !isSearching && (
              <div className="pt-1">
                {isLoading ? (
                  <Text className="animate-pulse p-2">Loading folder structure...</Text>
                ) : (
                  <Text className="p-2">Open a folder to see its structure.</Text>
                )}
              </div>
            )}
          </div>

          {googlePhotosEnabled && (
            <button
              type="button"
              className={clsx(
                'mt-2 flex items-center gap-2 p-1.5 rounded-md transition-colors cursor-pointer w-full text-left',
                selectedPath === GOOGLE_PHOTOS_FOLDER_PATH ? 'bg-surface' : 'hover:bg-card-active',
              )}
              onClick={onGooglePhotosSelect}
              data-tooltip={`Show ${googlePhotosAlbumTitle} in Google Photos`}
            >
              <Cloud size={16} className="shrink-0 text-text-secondary" />
              <span className="truncate select-none flex-1">Google Photos</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
