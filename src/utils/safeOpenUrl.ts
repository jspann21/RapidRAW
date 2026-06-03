import { open } from '@tauri-apps/plugin-shell';

interface OpenExternalUrlOptions {
  allowedHosts?: string[];
  allowHttp?: boolean;
}

export const normalizeExternalUrl = (value: string, options: OpenExternalUrlOptions = {}) => {
  const url = new URL(value);
  const allowedProtocols = options.allowHttp ? new Set(['https:', 'http:']) : new Set(['https:']);

  if (!allowedProtocols.has(url.protocol)) {
    throw new Error(`Blocked unsupported URL protocol: ${url.protocol}`);
  }

  if (url.username || url.password) {
    throw new Error('Blocked URL with embedded credentials.');
  }

  if (options.allowedHosts?.length) {
    const hostname = url.hostname.toLowerCase();
    const allowed = options.allowedHosts.some((host) => hostname === host.toLowerCase());
    if (!allowed) {
      throw new Error(`Blocked URL host: ${url.hostname}`);
    }
  }

  return url.toString();
};

export const openExternalUrl = async (value: string, options: OpenExternalUrlOptions = {}) => {
  await open(normalizeExternalUrl(value, options));
};
