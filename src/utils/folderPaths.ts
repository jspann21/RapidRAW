export const MAX_RECENT_FOLDERS = 10;

export const normalizeFolderPath = (path: string | null | undefined) =>
  (path || '').replace(/[\\/]+$/, '').replace(/\\/g, '/').toLowerCase();

export const isSameFolderPath = (a: string | null | undefined, b: string | null | undefined) =>
  normalizeFolderPath(a) === normalizeFolderPath(b);

export const getFolderDisplayName = (path: string) => {
  const parts = path.replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || path;
};

export const isPathWithinFolder = (path: string | null | undefined, folder: string | null | undefined) => {
  const normalizedPath = normalizeFolderPath(path);
  const normalizedFolder = normalizeFolderPath(folder);

  return (
    normalizedPath === normalizedFolder ||
    normalizedPath.startsWith(`${normalizedFolder}/`)
  );
};

export const replaceFolderPathPrefix = (path: string, oldPrefix: string, newPrefix: string) => {
  if (!isPathWithinFolder(path, oldPrefix)) {
    return path;
  }

  return `${newPrefix}${path.slice(oldPrefix.length)}`;
};

export const dedupeFolderPaths = (paths: string[]) => {
  const seen = new Set<string>();
  const result: string[] = [];

  paths.forEach((path) => {
    const normalized = normalizeFolderPath(path);
    if (!path || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    result.push(path);
  });

  return result;
};

export const nextRecentFolders = (recentFolders: string[] = [], openedPath: string, excludedFolders: string[] = []) => {
  const excludedSet = new Set(excludedFolders.map(normalizeFolderPath));
  const normalizedOpenedPath = normalizeFolderPath(openedPath);

  if (!openedPath || excludedSet.has(normalizedOpenedPath)) {
    return recentFolders.filter((path) => !excludedSet.has(normalizeFolderPath(path))).slice(0, MAX_RECENT_FOLDERS);
  }

  return dedupeFolderPaths([
    openedPath,
    ...recentFolders.filter(
      (path) => normalizeFolderPath(path) !== normalizedOpenedPath && !excludedSet.has(normalizeFolderPath(path)),
    ),
  ]).slice(0, MAX_RECENT_FOLDERS);
};
