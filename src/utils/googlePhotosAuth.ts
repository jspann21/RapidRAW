import { createElement, type MouseEvent } from 'react';
import { toast } from 'react-toastify';
import { useUIStore } from '../store/useUIStore';

export const GOOGLE_PHOTOS_REAUTH_REQUIRED_PREFIX = 'GOOGLE_PHOTOS_REAUTH_REQUIRED:';

export function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function isGooglePhotosReauthRequired(error: unknown) {
  const message = getErrorMessage(error);
  const lowerMessage = message.toLowerCase();
  return (
    message.includes(GOOGLE_PHOTOS_REAUTH_REQUIRED_PREFIX) ||
    lowerMessage.includes('invalid_grant') ||
    lowerMessage.includes('expired or revoked')
  );
}

export function formatGooglePhotosError(error: unknown) {
  const message = getErrorMessage(error).replace(GOOGLE_PHOTOS_REAUTH_REQUIRED_PREFIX, '').trim();
  return message || 'Google Photos authorization expired or was revoked.';
}

export function openGooglePhotosSettings() {
  useUIStore.getState().requestSettingsPanel('googlePhotos');
}

function googlePhotosSettingsToastContent(message: string) {
  return createElement('div', { className: 'space-y-2' }, [
    createElement('div', { key: 'message' }, message),
    createElement('div', { key: 'hint', className: 'text-sm font-semibold opacity-90' }, 'Open Google Photos Settings'),
  ]);
}

function googlePhotosReauthToastContent() {
  return createElement('div', { className: 'space-y-3' }, [
    createElement(
      'div',
      { key: 'message' },
      'Google Photos authorization expired or was revoked. Sign in again to continue syncing.',
    ),
    createElement(
      'button',
      {
        key: 'button',
        className:
          'rounded-md border border-current px-3 py-1 text-sm font-semibold text-inherit hover:bg-white/10 transition-colors',
        onClick: (event: MouseEvent<HTMLButtonElement>) => {
          event.stopPropagation();
          openGooglePhotosSettings();
        },
        type: 'button',
      },
      'Open Google Photos Settings',
    ),
  ]);
}

export function notifyGooglePhotosReauthRequired(toastId?: string | number) {
  const render = googlePhotosReauthToastContent();
  if (toastId !== undefined) {
    toast.update(toastId, {
      render,
      type: 'error',
      isLoading: false,
      autoClose: 10000,
      closeOnClick: false,
      onClick: openGooglePhotosSettings,
    });
    return;
  }

  toast.error(render, {
    autoClose: 10000,
    closeOnClick: false,
    onClick: openGooglePhotosSettings,
  });
}

export function notifyGooglePhotosSettingsRequired(message: string, toastId?: string | number) {
  const render = googlePhotosSettingsToastContent(message);

  if (toastId !== undefined) {
    toast.update(toastId, {
      render,
      type: 'error',
      isLoading: false,
      autoClose: 5000,
      closeOnClick: true,
      onClick: openGooglePhotosSettings,
    });
    return;
  }

  toast.error(render, {
    autoClose: 5000,
    closeOnClick: true,
    onClick: openGooglePhotosSettings,
  });
}
