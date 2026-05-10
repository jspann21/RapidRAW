import { create } from 'zustand';

export const GOOGLE_PHOTOS_FOLDER_PATH = 'rapidraw://google-photos';

export interface GooglePhotosSyncEntry {
  mediaItemId: string;
  productUrl?: string | null;
  baseUrl?: string | null;
  filename: string;
  syncedAt: string;
}

interface GooglePhotosState {
  syncIndex: Record<string, GooglePhotosSyncEntry>;
  isAlbumView: boolean;
  setGooglePhotos: (
    updater: Partial<GooglePhotosState> | ((state: GooglePhotosState) => Partial<GooglePhotosState>),
  ) => void;
  setSyncedEntries: (entries: Record<string, GooglePhotosSyncEntry>) => void;
  markSynced: (paths: string[], entries?: Record<string, GooglePhotosSyncEntry>) => void;
  markUnsynced: (paths: string[]) => void;
}

export const useGooglePhotosStore = create<GooglePhotosState>((set) => ({
  syncIndex: {},
  isAlbumView: false,

  setGooglePhotos: (updater) => set((state) => (typeof updater === 'function' ? updater(state) : updater)),

  setSyncedEntries: (entries) => set({ syncIndex: entries }),

  markSynced: (paths, entries = {}) =>
    set((state) => {
      const next = { ...state.syncIndex, ...entries };
      paths.forEach((path) => {
        if (!next[path]) {
          next[path] = {
            mediaItemId: '',
            filename: path.split(/[\\/]/).pop() || path,
            syncedAt: new Date().toISOString(),
          };
        }
      });
      return { syncIndex: next };
    }),

  markUnsynced: (paths) =>
    set((state) => {
      const next = { ...state.syncIndex };
      paths.forEach((path) => {
        delete next[path];
      });
      return { syncIndex: next };
    }),
}));
