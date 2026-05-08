export const normalizeDraggedImagePaths = (paths: string[]) => Array.from(new Set(paths)).filter(Boolean);
