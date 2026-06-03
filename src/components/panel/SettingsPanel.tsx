import { useEffect, useMemo, useState } from 'react';
import {
  ArrowLeft,
  Cloud,
  Cpu,
  Download,
  ExternalLink as ExternalLinkIcon,
  FileEdit,
  HardDrive,
  Server,
  Info,
  PlayCircle,
  RefreshCw,
  Trash2,
  Wifi,
  WifiOff,
  Zap,
  Plus,
  X,
  SlidersHorizontal,
  Keyboard,
  Bookmark,
  Scaling,
  Image as ImageIcon,
  Mouse,
  Touchpad,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { relaunch } from '@tauri-apps/plugin-process';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { Show, SignIn, useUser, useAuth, useClerk } from '@clerk/react';
import Button from '../ui/Button';
import ConfirmModal from '../modals/ConfirmModal';
import Dropdown, { OptionItem } from '../ui/Dropdown';
import Switch from '../ui/Switch';
import Input from '../ui/Input';
import Slider from '../ui/Slider';
import { ThemeProps, THEMES, DEFAULT_THEME_ID } from '../../utils/themes';
import { useTranslation } from 'react-i18next';
import { Invokes } from '../ui/AppProperties';
import {
  formatKeyCode,
  KeybindDefinition,
  KEYBIND_DEFINITIONS,
  KEYBIND_SECTIONS,
  normalizeCombo,
} from '../../utils/keyboardUtils';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useOsPlatform } from '../../hooks/useOsPlatform';
import { openExternalUrl } from '../../utils/safeOpenUrl';

interface ConfirmModalState {
  cancelText?: string;
  confirmText: string;
  confirmVariant: string;
  isOpen: boolean;
  message: string;
  onConfirm(): void;
  title: string;
}

interface DataActionItemProps {
  buttonAction(): void;
  buttonText: string;
  description: any;
  disabled?: boolean;
  icon: any;
  isProcessing: boolean;
  message: string;
  title: string;
}

interface KeybindRowProps {
  def: KeybindDefinition;
  currentCombo?: string[];
  osPlatform: string;
  onSave: (action: string, combo: string[]) => void;
  recordingAction: string | null;
  onStartRecording: (action: string) => void;
  isConflicting: boolean;
}

interface SettingItemProps {
  children: any;
  description?: string;
  label: string;
}

interface SettingsPanelProps {
  appSettings: any;
  initialCategory?: string;
  initialCategoryRequestId?: number;
  onBack(): void;
  onLibraryRefresh(): void;
  onSettingsChange(settings: any): Promise<void>;
  rootPaths: string[];
}

interface TestStatus {
  message: string;
  success: boolean | null;
  testing: boolean;
}

interface LocalAiModelInfo {
  id: string;
  name: string;
  filename: string;
  fileType: string;
  required: boolean;
  installed: boolean;
  valid: boolean;
  sizeBytes: number;
  sha256: string;
}

interface LocalAiStatus {
  isWindows: boolean;
  cudaAvailable: boolean;
  cudaProviderAvailable: boolean;
  cudaProviderError?: string | null;
  modelDir: string;
  modelDirWritable: boolean;
  modelDirError?: string | null;
  diskUsageBytes: number;
  requiredFileTypes: string[];
  runtimeDependencies: Array<{
    name: string;
    kind: string;
    found: boolean;
    path?: string | null;
  }>;
  missingRuntimeDependencies: string[];
  gpu: {
    name?: string | null;
    driverVersion?: string | null;
    vramMb?: number | null;
    computeCapability?: string | null;
    isNvidia: boolean;
  };
  models: LocalAiModelInfo[];
  localComfy: {
    runtimeDir: string;
    runtimeInstalled: boolean;
    customNodesInstalled: boolean;
    running: boolean;
    port?: number | null;
    generativeReady: boolean;
    lastError?: string | null;
  };
}

interface LocalAiDownloadProgress {
  modelName: string;
  downloadedBytes: number;
  totalBytes?: number | null;
}

type LocalAiTask =
  | 'runtime-refresh'
  | 'model-refresh'
  | 'download'
  | 'delete'
  | 'generative-delete'
  | 'self-test'
  | 'save-runtime'
  | 'generative-download'
  | 'runtime-download'
  | 'runtime-start'
  | 'runtime-stop'
  | 'runtime-delete'
  | 'generative-test';

interface LocalAiGenerationSettings {
  steps: number;
  cfg: number;
  samplerName: string;
  scheduler: string;
  denoise: number;
  cropTarget: number;
  maskBlendPixels: number;
  controlnetStrength: number;
  negativePrompt: string;
  seed: number | null;
}

interface MyLens {
  maker: string;
  model: string;
}

const EXECUTE_TIMEOUT = 3000;

const adjustmentVisibilityDefaults = {
  sharpening: true,
  presence: true,
  noiseReduction: true,
  chromaticAberration: false,
  vignette: true,
  colorCalibration: false,
  grain: true,
};

const resolutions: OptionItem<number>[] = [
  { value: 720, label: '720px' },
  { value: 1280, label: '1280px' },
  { value: 1920, label: '1920px' },
  { value: 2560, label: '2560px' },
  { value: 3840, label: '3840px' },
];

const thumbnailResolutions: OptionItem<number>[] = [
  { value: 640, label: '640px' },
  { value: 720, label: '720px' },
  { value: 960, label: '960px' },
  { value: 1080, label: '1080px' },
];

const zoomMultiplierOptions: OptionItem<number>[] = [
  { value: 1.0, label: '1.0x (Native)' },
  { value: 0.75, label: '0.75x' },
  { value: 0.5, label: '0.50x (Half)' },
  { value: 0.25, label: '0.25x' },
];

const formatBytes = (bytes: number) => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
};

const localAiGenerationDefaults: LocalAiGenerationSettings = {
  steps: 8,
  cfg: 1,
  samplerName: 'euler',
  scheduler: 'ddim_uniform',
  denoise: 1,
  cropTarget: 1280,
  maskBlendPixels: 32,
  controlnetStrength: 1,
  negativePrompt: 'blur, low quality, distortion, watermark',
  seed: null,
};

const localAiSamplerOptions: OptionItem<string>[] = [
  { value: 'euler', label: 'Euler' },
  { value: 'euler_ancestral', label: 'Euler Ancestral' },
  { value: 'dpmpp_2m', label: 'DPM++ 2M' },
  { value: 'dpmpp_sde', label: 'DPM++ SDE' },
  { value: 'dpmpp_2m_sde', label: 'DPM++ 2M SDE' },
];

const localAiSchedulerOptions: OptionItem<string>[] = [
  { value: 'ddim_uniform', label: 'DDIM Uniform' },
  { value: 'normal', label: 'Normal' },
  { value: 'karras', label: 'Karras' },
  { value: 'exponential', label: 'Exponential' },
  { value: 'simple', label: 'Simple' },
];

const GOOGLE_PHOTOS_LOGIN_POLL_INTERVAL_MS = 1500;
const GOOGLE_PHOTOS_LOGIN_MAX_POLLS = 60;
const GOOGLE_PHOTOS_BLOCKED_HINT_POLL = 8;

const isGooglePhotosErrorMessage = (message: string) => {
  const lowerMessage = message.toLowerCase();
  return ['blocked', 'denied', 'error', 'failed', 'not completed', 'timed out'].some((term) =>
    lowerMessage.includes(term),
  );
};
const KeybindRow = ({
  def,
  currentCombo,
  osPlatform,
  onSave,
  recordingAction,
  onStartRecording,
  isConflicting,
}: KeybindRowProps) => {
  const { t } = useTranslation();
  const recording = recordingAction === def.action;

  useEffect(() => {
    if (!recording) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onSave(def.action, []);
        onStartRecording('');
        return;
      }
      e.preventDefault();
      const parts = normalizeCombo(e, osPlatform);
      if (parts.length > 0 && !['ctrl', 'shift', 'alt'].includes(parts[parts.length - 1])) {
        onSave(def.action, parts);
        onStartRecording('');
      }
    };
    window.addEventListener('keydown', handler, { capture: true });
    return () => window.removeEventListener('keydown', handler, { capture: true });
  }, [recording, def.action, onSave, onStartRecording]);

  const displayCombo = currentCombo !== undefined ? (currentCombo.length ? currentCombo : null) : def.defaultCombo;

  return (
    <div className="flex justify-between items-center py-2">
      <Text variant={TextVariants.label}>{t(def.description as any)}</Text>
      <div className="flex items-center gap-1">
        {isConflicting && <span className="text-yellow-400 text-xs">⚠</span>}
        <button onClick={() => onStartRecording(def.action)} className="flex items-center gap-1 flex-wrap shrink-0">
          {recording ? (
            <Text
              as="kbd"
              variant={TextVariants.small}
              color={TextColors.accent}
              weight={TextWeights.semibold}
              className="px-2 py-1 font-sans bg-bg-primary border border-accent rounded-md animate-pulse"
            >
              {t('settings.controls.pressKey')}
            </Text>
          ) : (
            <Text
              as="kbd"
              variant={TextVariants.small}
              color={TextColors.primary}
              weight={TextWeights.semibold}
              className={`px-2 py-1 font-sans bg-bg-primary border rounded-md cursor-pointer hover:border-accent transition-colors ${isConflicting ? 'border-yellow-400' : 'border-border-color'}`}
            >
              {displayCombo ? (
                displayCombo.map((k) => formatKeyCode(k, osPlatform)).join(' + ')
              ) : (
                <span className="text-text-secondary italic">{t('settings.controls.notAssigned')}</span>
              )}
            </Text>
          )}
        </button>
      </div>
    </div>
  );
};

const SettingItem = ({ children, description, label }: SettingItemProps) => (
  <div>
    <Text variant={TextVariants.heading} className="block mb-2">
      {label}
    </Text>
    {children}
    {description && (
      <Text variant={TextVariants.small} className="mt-2">
        {description}
      </Text>
    )}
  </div>
);

const DataActionItem = ({
  buttonAction,
  buttonText,
  description,
  disabled = false,
  icon,
  isProcessing,
  message,
  title,
}: DataActionItemProps) => {
  const { t } = useTranslation();

  return (
    <div className="pb-8 border-b border-border-color last:border-b-0 last:pb-0">
      <Text variant={TextVariants.heading} className="mb-2">
        {title}
      </Text>
      <Text variant={TextVariants.small} className="mb-3">
        {description}
      </Text>
      <Button variant="destructive" onClick={buttonAction} disabled={isProcessing || disabled}>
        {icon}
        {isProcessing ? t('settings.data.statuses.processing') : buttonText}
      </Button>
      {message && (
        <Text color={TextColors.accent} className="mt-3">
          {message}
        </Text>
      )}
    </div>
  );
};

const CUDA_DOWNLOAD_URL = 'https://developer.nvidia.com/cuda-downloads';
const CUDNN_DOWNLOAD_URL = 'https://developer.nvidia.com/cudnn-downloads';
const CUDNN_WINDOWS_INSTALL_GUIDE_URL =
  'https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html';
const NVIDIA_DRIVER_DOWNLOAD_URL = 'https://www.nvidia.com/Download/index.aspx';

interface AiProviderSwitchProps {
  selectedProvider: string;
  onProviderChange: (provider: string) => void;
}

const AiProviderSwitch = ({ selectedProvider, onProviderChange }: AiProviderSwitchProps) => {
  const { t } = useTranslation();

  const aiProviders = useMemo(
    () => [
      { id: 'cpu', label: t('settings.processing.ai.providers.cpu'), icon: Cpu },
      { id: 'local-gpu', label: 'Local GPU', icon: Zap },
      { id: 'ai-connector', label: t('settings.processing.ai.providers.aiConnector'), icon: Server },
      //{ id: 'cloud', label: t('settings.processing.ai.providers.cloud'), icon: Cloud },
    ],
    [t],
  );

  return (
    <div className="relative flex w-full p-1 bg-bg-primary rounded-md border border-border-color">
      {aiProviders.map((provider) => (
        <button
          key={provider.id}
          onClick={() => onProviderChange(provider.id)}
          className={clsx(
            'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
            {
              'text-text-primary hover:bg-surface': selectedProvider !== provider.id,
              'text-button-text': selectedProvider === provider.id,
            },
          )}
          style={{ WebkitTapHighlightColor: 'transparent' }}
        >
          {selectedProvider === provider.id && (
            <motion.span
              layoutId="ai-provider-switch-bubble"
              className="absolute inset-0 z-0 bg-accent"
              style={{ borderRadius: 6 }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10 flex items-center">
            <provider.icon size={16} className="mr-2" />
            {provider.label}
          </span>
        </button>
      ))}
    </div>
  );
};

const CloudDashboard = () => {
  const { user } = useUser();
  const { getToken } = useAuth();
  const { signOut } = useClerk();
  const [usage, setUsage] = useState<{ requests: number; limit: number; month: string } | null>(null);
  const { t } = useTranslation();

  useEffect(() => {
    const fetchUsage = async () => {
      try {
        const token = await getToken();
        if (!token) return;
        const res = await fetch('https://getrapidraw.com/api/usage', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          setUsage(await res.json());
        }
      } catch (e) {
        console.error('Failed to fetch cloud usage', e);
      }
    };
    fetchUsage();
  }, [getToken]);

  const isPro = user?.publicMetadata?.plan === 'pro';

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between border-b border-border-color pb-4">
        <div className="flex items-center gap-3">
          <div>
            <Text variant={TextVariants.heading}>{user?.fullName || user?.primaryEmailAddress?.emailAddress}</Text>
            <Text variant={TextVariants.small} color={isPro ? TextColors.success : TextColors.error}>
              {isPro
                ? t('settings.processing.ai.cloud.signedIn.active')
                : t('settings.processing.ai.cloud.signedIn.inactive')}
            </Text>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            className="bg-transparent text-text-secondary hover:text-text-primary hover:bg-surface border-none shadow-none"
            onClick={() =>
              void openExternalUrl('https://www.getrapidraw.com/dashboard', {
                allowedHosts: ['www.getrapidraw.com'],
              })
            }
          >
            {t('settings.processing.ai.cloud.signedIn.manage')} <ExternalLinkIcon size={14} className="ml-1" />
          </Button>
          <Button
            variant="ghost"
            onClick={async () => {
              await signOut();
            }}
          >
            {t('settings.processing.ai.cloud.signedIn.logout')}
          </Button>
        </div>
      </div>

      {isPro ? (
        <div className="bg-surface p-4 rounded-md">
          <div className="flex justify-between items-center mb-2">
            <Text variant={TextVariants.label}>{t('settings.processing.ai.cloud.signedIn.usage')}</Text>
            <Text variant={TextVariants.small}>
              {t('settings.processing.ai.cloud.signedIn.usageStats', {
                requests: usage?.requests ?? 0,
                limit: usage?.limit ?? 500,
              })}
            </Text>
          </div>
          <div className="w-full bg-bg-primary rounded-full h-2">
            <div
              className="bg-accent h-2 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, ((usage?.requests ?? 0) / (usage?.limit ?? 500)) * 100)}%` }}
            />
          </div>
        </div>
      ) : (
        <div className="bg-red-900/10 border border-red-500/50 p-4 rounded-md text-center">
          <Text className="mb-3">{t('settings.processing.ai.cloud.signedOut.upgradeDesc')}</Text>
          <Button
            onClick={() =>
              void openExternalUrl('https://www.getrapidraw.com/cloud', {
                allowedHosts: ['www.getrapidraw.com'],
              })
            }
          >
            {t('settings.processing.ai.cloud.signedOut.upgradeBtn')}
          </Button>
        </div>
      )}
    </div>
  );
};

interface CanvasInputModeSwitchProps {
  mode: 'mouse' | 'trackpad';
  onModeChange: (mode: 'mouse' | 'trackpad') => void;
}

const CanvasInputModeSwitch = ({ mode, onModeChange }: CanvasInputModeSwitchProps) => {
  const { t } = useTranslation();

  const canvasInputModes = useMemo(
    () => [
      { id: 'mouse', label: t('settings.controls.modes.mouse'), icon: Mouse },
      { id: 'trackpad', label: t('settings.controls.modes.trackpad'), icon: Touchpad },
    ],
    [t],
  );

  return (
    <div className="relative flex w-full p-1 bg-bg-primary rounded-md border border-border-color">
      {canvasInputModes.map((item) => (
        <button
          key={item.id}
          onClick={() => onModeChange(item.id as 'mouse' | 'trackpad')}
          className={clsx(
            'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
            {
              'text-text-primary hover:bg-surface': mode !== item.id,
              'text-button-text': mode === item.id,
            },
          )}
          style={{ WebkitTapHighlightColor: 'transparent' }}
        >
          {mode === item.id && (
            <motion.span
              layoutId="canvas-input-mode-switch-bubble"
              className="absolute inset-0 z-0 bg-accent"
              style={{ borderRadius: 6 }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10 flex items-center">
            <item.icon size={16} className="mr-2" />
            {item.label}
          </span>
        </button>
      ))}
    </div>
  );
};

interface PreviewModeSwitchProps {
  mode: 'static' | 'dynamic';
  onModeChange: (mode: 'static' | 'dynamic') => void;
}

const PreviewModeSwitch = ({ mode, onModeChange }: PreviewModeSwitchProps) => {
  const { t } = useTranslation();

  const previewModes = useMemo(
    () => [
      { id: 'static', label: t('settings.processing.modes.static'), icon: ImageIcon },
      { id: 'dynamic', label: t('settings.processing.modes.dynamic'), icon: Scaling },
    ],
    [t],
  );

  return (
    <div className="relative flex w-full p-1 bg-bg-primary rounded-md border border-border-color">
      {previewModes.map((item) => (
        <button
          key={item.id}
          onClick={() => onModeChange(item.id as 'static' | 'dynamic')}
          className={clsx(
            'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
            {
              'text-text-primary hover:bg-surface': mode !== item.id,
              'text-button-text': mode === item.id,
            },
          )}
          style={{ WebkitTapHighlightColor: 'transparent' }}
        >
          {mode === item.id && (
            <motion.span
              layoutId="preview-mode-switch-bubble"
              className="absolute inset-0 z-0 bg-accent"
              style={{ borderRadius: 6 }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10 flex items-center">
            <item.icon size={16} className="mr-2" />
            {item.label}
          </span>
        </button>
      ))}
    </div>
  );
};

export default function SettingsPanel({
  appSettings,
  initialCategory,
  initialCategoryRequestId,
  onBack,
  onLibraryRefresh,
  onSettingsChange,
  rootPaths,
}: SettingsPanelProps) {
  const { user: _user } = useUser();
  const { t } = useTranslation();
  const [isClearing, setIsClearing] = useState(false);
  const [clearMessage, setClearMessage] = useState('');
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [cacheClearMessage, setCacheClearMessage] = useState('');
  const [isClearingAiTags, setIsClearingAiTags] = useState(false);
  const [aiTagsClearMessage, setAiTagsClearMessage] = useState('');
  const [isClearingTags, setIsClearingTags] = useState(false);
  const [tagsClearMessage, setTagsClearMessage] = useState('');
  const [confirmModalState, setConfirmModalState] = useState<ConfirmModalState>({
    confirmText: t('settings.data.modals.confirmClear'),
    confirmVariant: 'primary',
    isOpen: false,
    message: '',
    onConfirm: () => {},
    title: '',
  });
  const [testStatus, setTestStatus] = useState<TestStatus>({ message: '', success: null, testing: false });
  const [localAiStatus, setLocalAiStatus] = useState<LocalAiStatus | null>(null);
  const [localAiMessage, setLocalAiMessage] = useState('');
  const [localAiMessageScope, setLocalAiMessageScope] = useState<'generative' | 'lama' | null>(null);
  const [localAiDownloadProgress, setLocalAiDownloadProgress] = useState<LocalAiDownloadProgress | null>(null);
  const [localAiTask, setLocalAiTask] = useState<LocalAiTask | null>(null);
  const [localAiCudaRuntimePath, setLocalAiCudaRuntimePath] = useState(appSettings?.localAiCudaRuntimePath || '');
  const [localAiCudnnRuntimePath, setLocalAiCudnnRuntimePath] = useState(appSettings?.localAiCudnnRuntimePath || '');
  const [hasInteractedWithLivePreview, setHasInteractedWithLivePreview] = useState(false);
  const [recordingAction, setRecordingAction] = useState<string | null>(null);
  const [googlePhotosClientId, setGooglePhotosClientId] = useState(appSettings?.googlePhotosClientId || '');
  const [googlePhotosClientSecret, setGooglePhotosClientSecret] = useState(
    appSettings?.googlePhotosClientSecret || '',
  );
  const [googlePhotosAlbumTitleInput, setGooglePhotosAlbumTitleInput] = useState(
    appSettings?.googlePhotosAlbumTitle || 'RapidRaw',
  );
  const [googlePhotosBusy, setGooglePhotosBusy] = useState(false);
  const [googlePhotosMessage, setGooglePhotosMessage] = useState('');
  const [googlePhotosStatus, setGooglePhotosStatus] = useState<any>(null);

  const [aiProvider, setAiProvider] = useState(appSettings?.aiProvider || 'cpu');
  const [aiConnectorAddress, setAiConnectorAddress] = useState<string>(appSettings?.aiConnectorAddress || '');
  const [newShortcut, setNewShortcut] = useState('');
  const [newAiTag, setNewAiTag] = useState('');

  const [lensMakers, setLensMakers] = useState<string[]>([]);
  const [lensModels, setLensModels] = useState<string[]>([]);
  const [tempLensMaker, setTempLensMaker] = useState<string>('');
  const [tempLensModel, setTempLensModel] = useState<string>('');

  const osPlatform = useOsPlatform();
  const [processingSettings, setProcessingSettings] = useState({
    editorPreviewResolution: appSettings?.editorPreviewResolution || 1920,
    thumbnailResolution: appSettings?.thumbnailResolution || 720,
    rawHighlightCompression: appSettings?.rawHighlightCompression ?? 2.5,
    processingBackend: appSettings?.processingBackend || 'auto',
    linuxGpuOptimization: appSettings?.linuxGpuOptimization ?? false,
    highResZoomMultiplier: appSettings?.highResZoomMultiplier || 1.0,
    useFullDpiRendering: appSettings?.useFullDpiRendering ?? false,
    useWgpuRenderer:
      appSettings?.useWgpuRenderer ?? (osPlatform === 'linux' || osPlatform === 'android' ? false : true),
    thumbnailWorkerThreads: appSettings?.thumbnailWorkerThreads ?? 4,
    imageCacheSize: appSettings?.imageCacheSize ?? 5,
    rawPreprocessingColorNr: appSettings?.rawPreprocessingColorNr ?? 0.5,
    rawPreprocessingSharpening: appSettings?.rawPreprocessingSharpening ?? 0.35,
    applyPreprocessingToNonRaws: appSettings?.applyPreprocessingToNonRaws ?? false,
  });
  const [restartRequired, setRestartRequired] = useState(false);
  const [activeCategory, setActiveCategory] = useState(initialCategory || 'general');
  const [logPath, setLogPath] = useState<string | null>(null);
  const [logPathLoading, setLogPathLoading] = useState(true);
  const [logPathError, setLogPathError] = useState(false);
  const [dpr, setDpr] = useState(() => (typeof window !== 'undefined' ? window.devicePixelRatio : 1));

  const settingCategories = useMemo(
    () => [
      { id: 'general', label: t('settings.categories.general'), icon: SlidersHorizontal },
      { id: 'processing', label: t('settings.categories.processing'), icon: Cpu },
      { id: 'googlePhotos', label: 'Google Photos', icon: Cloud },
      { id: 'shortcuts', label: t('settings.categories.shortcuts'), icon: Keyboard },
    ],
    [t],
  );

  const livePreviewQualityOptions = useMemo<OptionItem<string>[]>(
    () => [
      { value: 'full', label: t('settings.processing.qualities.full') },
      { value: 'high', label: t('settings.processing.qualities.high') },
      { value: 'performance', label: t('settings.processing.qualities.performance') },
    ],
    [t],
  );

  const filteredBackendOptions = useMemo<OptionItem<string>[]>(() => {
    const rawOptions = [
      { value: 'auto', label: t('settings.processing.backends.auto') },
      { value: 'vulkan', label: t('settings.processing.backends.vulkan') },
      { value: 'dx12', label: t('settings.processing.backends.dx12') },
      { value: 'metal', label: t('settings.processing.backends.metal') },
      { value: 'gl', label: t('settings.processing.backends.gl') },
    ];
    return rawOptions.filter((opt) => {
      if (opt.value === 'metal' && osPlatform !== 'macos') return false;
      if (opt.value === 'dx12' && osPlatform === 'macos') return false;
      return true;
    });
  }, [t, osPlatform]);

  const linearRawOptions = useMemo<OptionItem<string>[]>(
    () => [
      { value: 'auto', label: t('settings.processing.preprocessing.linearOptions.auto') },
      { value: 'gamma', label: t('settings.processing.preprocessing.linearOptions.gamma') },
      { value: 'skip_calib', label: t('settings.processing.preprocessing.linearOptions.skip_calib') },
      { value: 'gamma_skip_calib', label: t('settings.processing.preprocessing.linearOptions.gamma_skip_calib') },
    ],
    [t],
  );

  const tonemapperOptions = useMemo<OptionItem<string>[]>(
    () => [
      { value: 'agx', label: t('settings.processing.preprocessing.tonemapperOptions.agx') },
      { value: 'basic', label: t('settings.processing.preprocessing.tonemapperOptions.basic') },
    ],
    [t],
  );

  const fontOptions = useMemo<OptionItem<string>[]>(
    () => [
      { value: 'poppins', label: t('settings.general.poppins') },
      { value: 'system', label: t('settings.general.system') },
    ],
    [t],
  );

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateDpr = () => setDpr(window.devicePixelRatio);

    const mediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    mediaQuery.addEventListener('change', updateDpr);

    window.addEventListener('resize', updateDpr);

    return () => {
      mediaQuery.removeEventListener('change', updateDpr);
      window.removeEventListener('resize', updateDpr);
    };
  }, []);

  useEffect(() => {
    if (initialCategory && settingCategories.some((category) => category.id === initialCategory)) {
      setActiveCategory(initialCategory);
    }
  }, [initialCategory, initialCategoryRequestId]);

  const customAiTags = Array.from(new Set<string>(appSettings?.customAiTags || []));
  const taggingShortcuts = Array.from(new Set<string>(appSettings?.taggingShortcuts || []));

  useEffect(() => {
    if (appSettings?.aiConnectorAddress !== aiConnectorAddress) {
      setAiConnectorAddress(appSettings?.aiConnectorAddress || '');
    }
    if (appSettings?.aiProvider !== aiProvider) {
      setAiProvider(appSettings?.aiProvider || 'cpu');
    }
    if (appSettings?.localAiCudaRuntimePath !== localAiCudaRuntimePath) {
      setLocalAiCudaRuntimePath(appSettings?.localAiCudaRuntimePath || '');
    }
    if (appSettings?.localAiCudnnRuntimePath !== localAiCudnnRuntimePath) {
      setLocalAiCudnnRuntimePath(appSettings?.localAiCudnnRuntimePath || '');
    }
    if (appSettings?.googlePhotosClientId !== googlePhotosClientId) {
      setGooglePhotosClientId(appSettings?.googlePhotosClientId || '');
    }
    if (appSettings?.googlePhotosClientSecret !== googlePhotosClientSecret) {
      setGooglePhotosClientSecret(appSettings?.googlePhotosClientSecret || '');
    }
    if (appSettings?.googlePhotosAlbumTitle !== googlePhotosAlbumTitleInput) {
      setGooglePhotosAlbumTitleInput(appSettings?.googlePhotosAlbumTitle || 'RapidRaw');
    }
    setProcessingSettings({
      editorPreviewResolution: appSettings?.editorPreviewResolution || 1920,
      thumbnailResolution: appSettings?.thumbnailResolution || 720,
      rawHighlightCompression: appSettings?.rawHighlightCompression ?? 2.5,
      processingBackend: appSettings?.processingBackend || 'auto',
      linuxGpuOptimization: appSettings?.linuxGpuOptimization ?? false,
      highResZoomMultiplier: appSettings?.highResZoomMultiplier || 1.0,
      useFullDpiRendering: appSettings?.useFullDpiRendering ?? false,
      useWgpuRenderer: appSettings?.useWgpuRenderer ?? true,
      thumbnailWorkerThreads: appSettings?.thumbnailWorkerThreads ?? 4,
      imageCacheSize: appSettings?.imageCacheSize ?? 5,
      rawPreprocessingColorNr: appSettings?.rawPreprocessingColorNr ?? 0.5,
      rawPreprocessingSharpening: appSettings?.rawPreprocessingSharpening ?? 0.35,
      applyPreprocessingToNonRaws: appSettings?.applyPreprocessingToNonRaws ?? false,
    });
    setRestartRequired(false);
  }, [appSettings]);

  useEffect(() => {
    const fetchLogPath = async () => {
      try {
        const path: string = await invoke(Invokes.GetLogFilePath);
        setLogPath(path);
      } catch (error) {
        console.error('Failed to get log file path:', error);
        setLogPathError(true);
      } finally {
        setLogPathLoading(false);
      }
    };
    fetchLogPath();
  }, []);

  const handleProcessingSettingChange = async (key: string, value: any) => {
    setProcessingSettings((prev) => ({ ...prev, [key]: value }));

    if (
      key === 'processingBackend' ||
      key === 'linuxGpuOptimization' ||
      key === 'useWgpuRenderer' ||
      key === 'thumbnailWorkerThreads'
    ) {
      setRestartRequired(true);
    } else {
      await onSettingsChange({ ...appSettings, [key]: value });
      if (
        key === 'rawHighlightCompression' ||
        key === 'rawPreprocessingColorNr' ||
        key === 'rawPreprocessingSharpening' ||
        key === 'applyPreprocessingToNonRaws'
      ) {
        await invoke('clear_image_caches');
      }
    }
  };

  const handleSaveAndRelaunch = async () => {
    await onSettingsChange({
      ...appSettings,
      ...processingSettings,
    });
    await relaunch();
  };

  const handleProviderChange = (provider: string) => {
    setAiProvider(provider);
    onSettingsChange({ ...appSettings, aiProvider: provider });
  };

  const localAiGenerationSettings: LocalAiGenerationSettings = {
    ...localAiGenerationDefaults,
    ...(appSettings?.localAiGenerationSettings || {}),
    seed: appSettings?.localAiGenerationSettings?.seed ?? null,
  };

  const updateLocalAiGenerationSettings = (patch: Partial<LocalAiGenerationSettings>) => {
    onSettingsChange({
      ...appSettings,
      localAiGenerationSettings: {
        ...localAiGenerationSettings,
        ...patch,
      },
    });
  };

  const resetLocalAiGenerationSettings = () => {
    onSettingsChange({
      ...appSettings,
      localAiGenerationSettings: localAiGenerationDefaults,
    });
  };

  const refreshLocalAiStatus = async (probeRuntime = false, task?: LocalAiTask) => {
    if (task) {
      setLocalAiTask(task);
      setLocalAiMessage(probeRuntime ? 'Checking CUDA runtime and model...' : 'Refreshing local model status...');
    }
    try {
      const status = await invoke<LocalAiStatus>(Invokes.GetLocalAiStatus, { probeRuntime });
      setLocalAiStatus(status);
      setLocalAiMessage('');
    } catch (err: unknown) {
      setLocalAiMessage(`Status failed: ${err}`);
    } finally {
      if (task) {
        setLocalAiTask(null);
      }
    }
  };

  useEffect(() => {
    if (aiProvider === 'local-gpu') {
      refreshLocalAiStatus(false);
    }
  }, [aiProvider]);

  useEffect(() => {
    const unlisten = listen<LocalAiDownloadProgress>('ai-model-download-progress', (event) => {
      if (event.payload?.modelName) {
        setLocalAiDownloadProgress(event.payload);
      }
    });

    return () => {
      unlisten.then((cleanup) => cleanup());
    };
  }, []);

  const handleDownloadLocalAiModel = async () => {
    setLocalAiDownloadProgress({ modelName: 'LaMa Inpainting', downloadedBytes: 0, totalBytes: null });
    setLocalAiMessageScope('lama');
    setLocalAiTask('download');
    setLocalAiMessage('Downloading LaMa inpainting model...');
    try {
      await invoke(Invokes.DownloadLocalAiModel, { modelId: 'lama-inpainting' });
      setLocalAiMessage('Model downloaded and verified.');
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Download failed: ${err}`);
    } finally {
      setLocalAiTask(null);
      setLocalAiDownloadProgress(null);
    }
  };

  const handleDeleteLocalAiModel = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('lama');
    setLocalAiTask('delete');
    setLocalAiMessage('Deleting local model...');
    try {
      await invoke(Invokes.DeleteLocalAiModel, { modelId: 'lama-inpainting' });
      setLocalAiMessage('Model deleted.');
      await refreshLocalAiStatus(false);
    } catch (err: unknown) {
      setLocalAiMessage(`Delete failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleRunLocalAiSelfTest = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('lama');
    setLocalAiTask('self-test');
    setLocalAiMessage('Running CUDA self-test...');
    try {
      const result = await invoke<string>(Invokes.RunLocalAiSelfTest);
      setLocalAiMessage(result);
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Self-test failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleDownloadLocalAiRuntime = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('generative');
    setLocalAiTask('runtime-download');
    setLocalAiMessage('Downloading Local GPU SDXL runtime...');
    try {
      await invoke(Invokes.DownloadLocalAiRuntime);
      setLocalAiMessage('Local GPU SDXL runtime installed.');
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Runtime install failed: ${err}`);
    } finally {
      setLocalAiTask(null);
      setLocalAiDownloadProgress(null);
    }
  };

  const handleDownloadLocalAiGenerativeAssets = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('generative');
    setLocalAiTask('generative-download');
    setLocalAiMessage('Downloading Local GPU SDXL models...');
    try {
      await invoke(Invokes.DownloadLocalAiGenerativeAssets);
      setLocalAiMessage('Local GPU SDXL models downloaded and verified.');
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`SDXL model download failed: ${err}`);
    } finally {
      setLocalAiTask(null);
      setLocalAiDownloadProgress(null);
    }
  };

  const handleDeleteLocalAiGenerativeAssets = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('generative');
    setLocalAiTask('generative-delete');
    setLocalAiMessage('Deleting Local GPU SDXL models...');
    try {
      for (const model of localAiGenerativeModels) {
        if (model.installed) {
          await invoke(Invokes.DeleteLocalAiModel, { modelId: model.id });
        }
      }
      setLocalAiMessage('Local GPU SDXL models deleted.');
      await refreshLocalAiStatus(false);
    } catch (err: unknown) {
      setLocalAiMessage(`SDXL model delete failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleStartLocalAiRuntime = async () => {
    setLocalAiMessageScope('generative');
    setLocalAiTask('runtime-start');
    setLocalAiMessage('Starting Local GPU SDXL runtime...');
    try {
      await invoke(Invokes.StartLocalAiRuntime);
      setLocalAiMessage('Local GPU SDXL runtime is running.');
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Runtime start failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleStopLocalAiRuntime = async () => {
    setLocalAiMessageScope('generative');
    setLocalAiTask('runtime-stop');
    setLocalAiMessage('Stopping Local GPU SDXL runtime...');
    try {
      await invoke(Invokes.StopLocalAiRuntime);
      setLocalAiMessage('Local GPU SDXL runtime stopped.');
      await refreshLocalAiStatus(false);
    } catch (err: unknown) {
      setLocalAiMessage(`Runtime stop failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleDeleteLocalAiRuntime = async () => {
    setLocalAiDownloadProgress(null);
    setLocalAiMessageScope('generative');
    setLocalAiTask('runtime-delete');
    setLocalAiMessage('Deleting Local GPU SDXL runtime...');
    try {
      await invoke(Invokes.DeleteLocalAiRuntime);
      setLocalAiMessage('Local GPU SDXL runtime deleted.');
      await refreshLocalAiStatus(false);
    } catch (err: unknown) {
      setLocalAiMessage(`Runtime delete failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleRunLocalGenerativeSelfTest = async () => {
    setLocalAiMessageScope('generative');
    setLocalAiTask('generative-test');
    setLocalAiMessage('Running Local GPU SDXL self-test...');
    try {
      const result = await invoke<string>(Invokes.RunLocalGenerativeSelfTest);
      setLocalAiMessage(result);
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Generative self-test failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handleLocalAiRuntimePathSave = async () => {
    setLocalAiMessageScope('generative');
    setLocalAiTask('save-runtime');
    setLocalAiMessage('Saving CUDA runtime paths...');
    try {
      await onSettingsChange({
        ...appSettings,
        localAiCudaRuntimePath: localAiCudaRuntimePath.trim() || undefined,
        localAiCudnnRuntimePath: localAiCudnnRuntimePath.trim() || undefined,
      });
      await refreshLocalAiStatus(true);
    } catch (err: unknown) {
      setLocalAiMessage(`Save failed: ${err}`);
    } finally {
      setLocalAiTask(null);
    }
  };

  const handlePreviewModeChange = (mode: 'static' | 'dynamic') => {
    const enableZoomHifi = mode === 'dynamic';
    onSettingsChange({ ...appSettings, enableZoomHifi });
  };

  const handleTempMakerChange = (maker: string) => {
    setTempLensMaker(maker);
    setTempLensModel('');
    setLensModels([]);
    if (maker) {
      invoke('get_lensfun_lenses_for_maker', { maker })
        .then((l: any) => setLensModels(l))
        .catch(console.error);
    }
  };

  const handleAddLens = () => {
    if (tempLensMaker && tempLensModel) {
      const currentLenses: MyLens[] = appSettings?.myLenses || [];
      if (!currentLenses.some((l) => l.maker === tempLensMaker && l.model === tempLensModel)) {
        const newLenses = [...currentLenses, { maker: tempLensMaker, model: tempLensModel }];

        newLenses.sort((a, b) => {
          const makerComp = a.maker.localeCompare(b.maker);
          if (makerComp !== 0) return makerComp;
          return a.model.localeCompare(b.model);
        });

        onSettingsChange({
          ...appSettings,
          myLenses: newLenses,
        });
        setTempLensMaker('');
        setTempLensModel('');
        setLensModels([]);
      }
    }
  };

  const handleRemoveLens = (index: number) => {
    const currentLenses: MyLens[] = appSettings?.myLenses || [];
    const newLenses = [...currentLenses];
    newLenses.splice(index, 1);
    onSettingsChange({ ...appSettings, myLenses: newLenses });
  };

  const effectiveRootPaths = rootPaths?.length > 0 ? rootPaths : appSettings?.rootFolders || [];

  const executeClearSidecars = async () => {
    setIsClearing(true);
    setClearMessage(t('settings.data.statuses.deleting'));
    try {
      let totalCount = 0;
      for (const root of effectiveRootPaths) {
        const count: number = await invoke(Invokes.ClearAllSidecars, { rootPath: root });
        totalCount += count;
      }
      setClearMessage(t('settings.data.statuses.sidecarSuccess', { count: totalCount }));
      onLibraryRefresh();
    } catch (err: any) {
      console.error('Failed to clear sidecars:', err);
      setClearMessage(`Error: ${err}`);
    } finally {
      setTimeout(() => {
        setIsClearing(false);
        setClearMessage('');
      }, EXECUTE_TIMEOUT);
    }
  };

  const handleClearSidecars = () => {
    setConfirmModalState({
      confirmText: t('settings.data.modals.confirmDeleteAllEdits'),
      confirmVariant: 'destructive',
      isOpen: true,
      message: t('settings.data.modals.sidecarMessage'),
      onConfirm: executeClearSidecars,
      title: t('settings.data.modals.confirmTitle'),
    });
  };

  const executeClearAiTags = async () => {
    setIsClearingAiTags(true);
    setAiTagsClearMessage(t('settings.data.statuses.clearingAi'));
    try {
      let totalCount = 0;
      for (const root of effectiveRootPaths) {
        const count: number = await invoke(Invokes.ClearAiTags, { rootPath: root });
        totalCount += count;
      }
      setAiTagsClearMessage(t('settings.data.statuses.aiSuccess', { count: totalCount }));
      onLibraryRefresh();
    } catch (err: any) {
      console.error('Failed to clear AI tags:', err);
      setAiTagsClearMessage(`Error: ${err}`);
    } finally {
      setTimeout(() => {
        setIsClearingAiTags(false);
        setAiTagsClearMessage('');
      }, EXECUTE_TIMEOUT);
    }
  };

  const handleClearAiTags = () => {
    setConfirmModalState({
      confirmText: t('settings.data.modals.confirmClearAi'),
      confirmVariant: 'destructive',
      isOpen: true,
      message: t('settings.data.modals.aiMessage'),
      onConfirm: executeClearAiTags,
      title: t('settings.data.modals.confirmAiTitle'),
    });
  };

  const executeClearTags = async () => {
    setIsClearingTags(true);
    setTagsClearMessage(t('settings.data.statuses.clearingAll'));
    try {
      let totalCount = 0;
      for (const root of effectiveRootPaths) {
        const count: number = await invoke(Invokes.ClearAllTags, { rootPath: root });
        totalCount += count;
      }
      setTagsClearMessage(t('settings.data.statuses.allSuccess', { count: totalCount }));
      onLibraryRefresh();
    } catch (err: any) {
      console.error('Failed to clear tags:', err);
      setTagsClearMessage(`Error: ${err}`);
    } finally {
      setTimeout(() => {
        setIsClearingTags(false);
        setTagsClearMessage('');
      }, EXECUTE_TIMEOUT);
    }
  };

  const handleClearTags = () => {
    setConfirmModalState({
      confirmText: t('settings.data.modals.confirmClearAll'),
      confirmVariant: 'destructive',
      isOpen: true,
      message: t('settings.data.modals.allMessage'),
      onConfirm: executeClearTags,
      title: t('settings.data.modals.confirmAllTitle'),
    });
  };

  const shortcutTagVariants = {
    visible: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 500, damping: 30 } },
    exit: { opacity: 0, scale: 0.8, transition: { duration: 0.15 } },
  };

  const executeClearCache = async () => {
    setIsClearingCache(true);
    setCacheClearMessage(t('settings.data.statuses.clearingCache'));
    try {
      await invoke(Invokes.ClearThumbnailCache);
      setCacheClearMessage(t('settings.data.statuses.cacheSuccess'));
      onLibraryRefresh();
    } catch (err: any) {
      console.error('Failed to clear thumbnail cache:', err);
      setCacheClearMessage(`Error: ${err}`);
    } finally {
      setTimeout(() => {
        setIsClearingCache(false);
        setCacheClearMessage('');
      }, EXECUTE_TIMEOUT);
    }
  };

  const handleClearCache = () => {
    setConfirmModalState({
      confirmText: t('settings.data.modals.confirmClearCache'),
      confirmVariant: 'destructive',
      isOpen: true,
      message: t('settings.data.modals.cacheMessage'),
      onConfirm: executeClearCache,
      title: t('settings.data.modals.confirmCacheTitle'),
    });
  };

  const handleTestConnection = async () => {
    if (!aiConnectorAddress) {
      return;
    }
    setTestStatus({ testing: true, message: t('settings.processing.ai.connector.testing'), success: null });
    try {
      await invoke(Invokes.TestAIConnectorConnection, { address: aiConnectorAddress });
      setTestStatus({ testing: false, message: t('settings.processing.ai.connector.success'), success: true });
    } catch (err) {
      setTestStatus({ testing: false, message: t('settings.processing.ai.connector.failed'), success: false });
      console.error('AI Connector connection test failed:', err);
    } finally {
      setTimeout(() => setTestStatus({ testing: false, message: '', success: null }), EXECUTE_TIMEOUT);
    }
  };

  const closeConfirmModal = () => {
    setConfirmModalState({ ...confirmModalState, isOpen: false });
  };

  const handleAddShortcut = () => {
    const parsedTags = newShortcut
      .split(',')
      .map((t) => t.trim().toLowerCase())
      .filter((t) => t.length > 0);

    if (parsedTags.length > 0) {
      const uniqueShortcuts = Array.from(new Set([...taggingShortcuts, ...parsedTags])).sort();
      onSettingsChange({ ...appSettings, taggingShortcuts: uniqueShortcuts });
    }
    setNewShortcut('');
  };

  const handleRemoveShortcut = (shortcutToRemove: string) => {
    const uniqueShortcuts = taggingShortcuts.filter((s) => s !== shortcutToRemove);
    onSettingsChange({ ...appSettings, taggingShortcuts: uniqueShortcuts });
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddShortcut();
    }
  };

  const handleAddAiTag = () => {
    const parsedTags = newAiTag
      .split(',')
      .map((t) => t.trim().toLowerCase())
      .filter((t) => t.length > 0);

    if (parsedTags.length > 0) {
      const uniqueTags = Array.from(new Set([...customAiTags, ...parsedTags])).sort();
      onSettingsChange({ ...appSettings, customAiTags: uniqueTags });
    }
    setNewAiTag('');
  };

  const handleRemoveAiTag = (tagToRemove: string) => {
    const uniqueTags = customAiTags.filter((t) => t !== tagToRemove);
    onSettingsChange({ ...appSettings, customAiTags: uniqueTags });
  };

  const handleAiTagInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddAiTag();
    }
  };

  const handleKeybindSave = (action: string, combo: string[]) => {
    const newKeybinds = { ...(appSettings?.keybinds || {}), [action]: combo };
    onSettingsChange({ ...appSettings, keybinds: newKeybinds });
  };

  const refreshGooglePhotosStatus = async () => {
    try {
      const status = await invoke(Invokes.GooglePhotosGetStatus);
      setGooglePhotosStatus(status);
    } catch (err) {
      console.error('Failed to load Google Photos status:', err);
    }
  };

  const saveGooglePhotosCredentials = async (enabled = appSettings?.googlePhotosIntegrationEnabled ?? false) => {
    await onSettingsChange({
      ...appSettings,
      googlePhotosClientId,
      googlePhotosClientSecret,
      googlePhotosAlbumTitle: googlePhotosAlbumTitleInput || 'RapidRaw',
      googlePhotosIntegrationEnabled: enabled,
    });
  };

  const handleGooglePhotosLogin = async () => {
    setGooglePhotosBusy(true);
    setGooglePhotosMessage('Opening Google sign-in...');
    let loginState: string | null = null;
    try {
      await saveGooglePhotosCredentials(true);
      const start: any = await invoke(Invokes.GooglePhotosStartLogin);
      loginState = start.state;
      await openExternalUrl(start.authorizationUrl, { allowedHosts: ['accounts.google.com'] });

      for (let attempt = 0; attempt < GOOGLE_PHOTOS_LOGIN_MAX_POLLS; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, GOOGLE_PHOTOS_LOGIN_POLL_INTERVAL_MS));
        const poll: any = await invoke(Invokes.GooglePhotosPollLogin, { state: loginState });
        const waitingMessage =
          attempt >= GOOGLE_PHOTOS_BLOCKED_HINT_POLL
            ? 'Waiting for Google sign-in. If Google shows "Access blocked", add this Google account as an OAuth test user or complete OAuth verification in Google Cloud, then try again.'
            : poll.message;
        setGooglePhotosMessage(poll.complete ? poll.message : waitingMessage);
        if (poll.complete) {
          loginState = null;
          if (!poll.authenticated) {
            throw new Error(poll.message || 'Google sign-in did not complete.');
          }
          await onSettingsChange({
            ...appSettings,
            googlePhotosClientId,
            googlePhotosClientSecret,
            googlePhotosAlbumTitle: googlePhotosAlbumTitleInput || 'RapidRaw',
            googlePhotosIntegrationEnabled: true,
          });
          await refreshGooglePhotosStatus();
          return;
        }
      }
      if (loginState) {
        await invoke(Invokes.GooglePhotosCancelLogin, { state: loginState }).catch(() => {});
        loginState = null;
      }
      throw new Error(
        'Google sign-in timed out. If the browser shows "Access blocked", Google did not return to RapidRAW. Add your account as an OAuth test user or complete OAuth verification, then try again.',
      );
    } catch (err: any) {
      if (loginState) {
        await invoke(Invokes.GooglePhotosCancelLogin, { state: loginState }).catch(() => {});
      }
      setGooglePhotosMessage(err?.message || String(err));
    } finally {
      setGooglePhotosBusy(false);
    }
  };

  const executeDisableGooglePhotosIntegration = async () => {
    await onSettingsChange({ ...appSettings, googlePhotosIntegrationEnabled: false });
    const savedAlbumTitle = appSettings?.googlePhotosAlbumTitle || googlePhotosAlbumTitleInput || 'RapidRaw';
    setGooglePhotosMessage(
      appSettings?.googlePhotosAlbumId
        ? `Google Photos sync is off. The album "${savedAlbumTitle}" was not deleted and will be reused if you turn sync back on.`
        : 'Google Photos sync is off. Saved Google Photos settings were kept.',
    );
  };

  const handleGooglePhotosIntegrationToggle = (checked: boolean) => {
    if (checked) {
      onSettingsChange({ ...appSettings, googlePhotosIntegrationEnabled: true });
      if (appSettings?.googlePhotosAlbumId) {
        const savedAlbumTitle = appSettings?.googlePhotosAlbumTitle || googlePhotosAlbumTitleInput || 'RapidRaw';
        setGooglePhotosMessage(`Google Photos sync is on. RapidRAW will use "${savedAlbumTitle}".`);
      } else {
        setGooglePhotosMessage('Google Photos sync is on. Sign in and create an album before syncing photos.');
      }
      return;
    }

    if (!appSettings?.googlePhotosIntegrationEnabled) {
      return;
    }

    const savedAlbumTitle = appSettings?.googlePhotosAlbumTitle || googlePhotosAlbumTitleInput || 'RapidRaw';
    const albumLine = appSettings?.googlePhotosAlbumId
      ? `\n\nSaved album: "${savedAlbumTitle}".`
      : '';
    setConfirmModalState({
      cancelText: 'Keep Sync On',
      confirmText: 'Turn Off Sync',
      confirmVariant: 'destructive',
      isOpen: true,
      message:
        `RapidRAW will stop syncing to Google Photos and hide the Google Photos album from the folder sidebar.${albumLine}\n\nThis does not delete the album or photos from Google Photos. Your Google Photos credentials, album ID, and album title stay saved locally, so turning Google Photos back on will reuse the same album settings.`,
      onConfirm: () => {
        void executeDisableGooglePhotosIntegration();
      },
      title: 'Turn Off Google Photos Sync?',
    });
  };

  const handleGooglePhotosCreateAlbum = async () => {
    setGooglePhotosBusy(true);
    setGooglePhotosMessage('Creating Google Photos album...');
    try {
      await saveGooglePhotosCredentials(true);
      const album: any = await invoke(Invokes.GooglePhotosCreateAlbum, {
        title: googlePhotosAlbumTitleInput || 'RapidRaw',
      });
      await onSettingsChange({
        ...appSettings,
        googlePhotosClientId,
        googlePhotosClientSecret,
        googlePhotosIntegrationEnabled: true,
        googlePhotosAlbumId: album.id,
        googlePhotosAlbumTitle: album.title,
      });
      setGooglePhotosAlbumTitleInput(album.title);
      setGooglePhotosMessage(`Using album "${album.title}".`);
      await refreshGooglePhotosStatus();
    } catch (err: any) {
      setGooglePhotosMessage(err?.message || String(err));
    } finally {
      setGooglePhotosBusy(false);
    }
  };

  const handleGooglePhotosRenameAlbum = async () => {
    setGooglePhotosBusy(true);
    setGooglePhotosMessage('Renaming Google Photos album...');
    try {
      const album: any = await invoke(Invokes.GooglePhotosRenameAlbum, {
        title: googlePhotosAlbumTitleInput || 'RapidRaw',
      });
      await onSettingsChange({
        ...appSettings,
        googlePhotosClientId,
        googlePhotosClientSecret,
        googlePhotosAlbumId: album.id,
        googlePhotosAlbumTitle: album.title,
      });
      setGooglePhotosMessage(`Renamed album to "${album.title}".`);
      await refreshGooglePhotosStatus();
    } catch (err: any) {
      setGooglePhotosMessage(err?.message || String(err));
    } finally {
      setGooglePhotosBusy(false);
    }
  };

  const handleGooglePhotosDisconnect = async () => {
    setGooglePhotosBusy(true);
    setGooglePhotosMessage('Disconnecting Google Photos...');
    try {
      await invoke(Invokes.GooglePhotosDisconnect);
      await onSettingsChange({ ...appSettings, googlePhotosIntegrationEnabled: false });
      setGooglePhotosStatus(null);
      setGooglePhotosMessage('Google Photos disconnected.');
    } catch (err: any) {
      setGooglePhotosMessage(err?.message || String(err));
    } finally {
      setGooglePhotosBusy(false);
    }
  };

  useEffect(() => {
    if (activeCategory === 'googlePhotos') {
      refreshGooglePhotosStatus();
    }
  }, [activeCategory]);

  const conflictingKeys = useMemo(() => {
    const map = new Map<string, Set<string>>();
    const userKb = appSettings?.keybinds || {};
    for (const def of KEYBIND_DEFINITIONS) {
      const userCombo = userKb[def.action];
      const effective = userCombo?.length ? userCombo : userCombo === undefined ? def.defaultCombo : null;
      if (!effective) continue;
      const key = effective.join('+');
      if (!map.has(key)) map.set(key, new Set());
      map.get(key)!.add(def.action);
    }
    const keys = new Set<string>();
    for (const [, actions] of map) {
      if (actions.size > 1) actions.forEach((k) => keys.add(k));
    }
    return keys;
  }, [appSettings?.keybinds]);

  const localAiModel = localAiStatus?.models.find((model) => model.id === 'lama-inpainting');
  const localAiGenerativeModels = (localAiStatus?.models || []).filter((model) => model.id.startsWith('comfy-'));
  const localAiGenerativeModelsReady =
    localAiGenerativeModels.length > 0 && localAiGenerativeModels.every((model) => model.installed && model.valid);
  const localAiGenerativeTask =
    localAiTask === 'runtime-download' ||
    localAiTask === 'generative-download' ||
    localAiTask === 'runtime-start' ||
    localAiTask === 'runtime-stop' ||
    localAiTask === 'runtime-delete' ||
    localAiTask === 'generative-delete' ||
    localAiTask === 'generative-test';
  const localAiLamaTask =
    localAiTask === 'model-refresh' || localAiTask === 'download' || localAiTask === 'delete' || localAiTask === 'self-test';
  const localAiDownloadTotalBytes = localAiDownloadProgress?.totalBytes || 0;
  const localAiDownloadPercent = localAiDownloadTotalBytes
    ? Math.min(100, Math.round((localAiDownloadProgress!.downloadedBytes / localAiDownloadTotalBytes) * 100))
    : null;
  const localAiDownloadLabel =
    localAiDownloadProgress && localAiDownloadTotalBytes
      ? `Downloading ${formatBytes(localAiDownloadProgress.downloadedBytes)} of ${formatBytes(localAiDownloadTotalBytes)}`
      : localAiDownloadProgress
        ? `Downloading ${formatBytes(localAiDownloadProgress.downloadedBytes)}`
        : '';
  const isLocalAiBusy = localAiTask !== null;
  const lowerLocalAiMessage = localAiMessage.toLowerCase();
  const isLocalAiGenerativeMessage = !!localAiMessage && localAiMessageScope === 'generative';
  const isLocalAiLamaMessage = !!localAiMessage && localAiMessageScope === 'lama';
  const localAiLamaDownloadButtonLabel =
    localAiTask === 'download'
      ? localAiDownloadPercent === null
        ? localAiDownloadProgress
          ? 'Downloading'
          : 'Download'
        : `Downloading ${localAiDownloadPercent}%`
      : 'Download';
  const localAiTaskLabel =
    localAiTask === 'runtime-refresh'
      ? 'Checking CUDA runtime...'
      : localAiTask === 'model-refresh'
        ? 'Refreshing model status...'
        : localAiTask === 'delete'
          ? 'Deleting local model...'
          : localAiTask === 'self-test'
            ? 'Running CUDA self-test...'
            : localAiTask === 'save-runtime'
              ? 'Saving runtime paths...'
              : localAiTask === 'runtime-download'
                ? 'Installing SDXL runtime...'
                : localAiTask === 'generative-download'
                  ? 'Downloading SDXL models...'
                  : localAiTask === 'generative-delete'
                    ? 'Deleting SDXL models...'
                    : localAiTask === 'runtime-start'
                      ? 'Starting SDXL runtime...'
                      : localAiTask === 'runtime-stop'
                        ? 'Stopping SDXL runtime...'
                        : localAiTask === 'runtime-delete'
                          ? 'Deleting SDXL runtime...'
                          : localAiTask === 'generative-test'
                            ? 'Running SDXL self-test...'
                            : '';
  const secondaryLocalAiButtonClass =
    'bg-surface text-text-primary border border-border-color hover:bg-bg-primary disabled:text-text-secondary';
  const primaryLocalAiButtonClass = '';
  const localAiGroupedButtonClass = 'h-10 w-44 justify-start whitespace-nowrap';
  const localAiIconButtonClass = `${secondaryLocalAiButtonClass} h-10 w-11 px-0`;
  const localAiRuntimeDependencies = localAiStatus?.runtimeDependencies || [];
  const missingLocalAiRuntimeDependencies = localAiStatus?.missingRuntimeDependencies || [];
  const missingCudaRuntime = missingLocalAiRuntimeDependencies.some((dependency) => {
    const lower = dependency.toLowerCase();
    return lower.includes('cuda') || lower.includes('cublas');
  });
  const missingCudnnRuntime = missingLocalAiRuntimeDependencies.some((dependency) =>
    dependency.toLowerCase().includes('cudnn'),
  );
  const cudaRuntimeReady =
    localAiRuntimeDependencies.some((dependency) => dependency.kind === 'CUDA' && dependency.found) &&
    !missingCudaRuntime;
  const cudnnRuntimeReady =
    localAiRuntimeDependencies.some((dependency) => dependency.kind === 'cuDNN' && dependency.found) &&
    !missingCudnnRuntime;
  const runtimeStatusLabel = missingCudnnRuntime
    ? 'cuDNN 9 missing'
    : missingCudaRuntime
      ? 'CUDA Toolkit missing'
      : localAiStatus?.cudaProviderAvailable
        ? 'Runtime tested'
        : 'Runtime files found';
  const localAiProviderFailed =
    !!localAiStatus?.cudaProviderError && missingLocalAiRuntimeDependencies.length === 0;
  const localAiGenerativeReady =
    !!localAiStatus?.isWindows &&
    !!localAiStatus?.cudaAvailable &&
    !!localAiStatus?.localComfy?.generativeReady &&
    localAiGenerativeModelsReady;
  const localAiPrerequisitesReady =
    !!localAiStatus?.isWindows &&
    !!localAiStatus?.cudaAvailable &&
    !!localAiStatus?.modelDirWritable &&
    (localAiStatus?.missingRuntimeDependencies.length || 0) === 0 &&
    !!localAiModel?.installed &&
    !!localAiModel?.valid;
  const localAiReady = localAiPrerequisitesReady && !localAiProviderFailed;
  const localAiStatusMessage = localAiReady
    ? localAiStatus?.cudaProviderAvailable
      ? 'Local GPU is ready.'
      : 'Local GPU is ready. Run Test is optional.'
    : localAiProviderFailed
      ? 'CUDA runtime check failed. Refresh or run the test after fixing the error below.'
      : !localAiStatus?.cudaAvailable
        ? 'No NVIDIA CUDA GPU detected.'
        : !localAiModel?.installed
          ? 'Download the LaMa model to enable Local GPU inpainting.'
          : missingLocalAiRuntimeDependencies.length
            ? 'Install the missing CUDA runtime shown above.'
            : 'Local GPU setup is incomplete.';
  const googlePhotosIntegrationEnabled = appSettings?.googlePhotosIntegrationEnabled ?? false;
  const savedGooglePhotosAlbumTitle = appSettings?.googlePhotosAlbumTitle || googlePhotosAlbumTitleInput || 'RapidRaw';
  const hasSavedGooglePhotosAlbum = Boolean(appSettings?.googlePhotosAlbumId);

  return (
    <>
      <ConfirmModal {...confirmModalState} onClose={closeConfirmModal} />
      <div className="flex flex-col h-full w-full text-text-primary">
        <header className="shrink-0 flex flex-wrap items-center justify-between gap-y-4 mb-8 pt-4">
          <div className="flex items-center shrink-0">
            <Button
              className="mr-4 hover:bg-surface text-text-primary rounded-full"
              onClick={onBack}
              size="icon"
              variant="ghost"
              data-tooltip={t('settings.tooltips.goHome')}
            >
              <ArrowLeft />
            </Button>
            <Text variant={TextVariants.display} color={TextColors.accent} className="whitespace-nowrap">
              {t('settings.title')}
            </Text>
          </div>

          <div className="relative flex w-full min-[1200px]:w-150 p-2 bg-surface rounded-md">
            {settingCategories.map((category) => (
              <button
                key={category.id}
                onClick={() => setActiveCategory(category.id)}
                className={clsx(
                  'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
                  {
                    'text-text-primary hover:bg-surface': activeCategory !== category.id,
                    'text-button-text': activeCategory === category.id,
                  },
                )}
                style={{ WebkitTapHighlightColor: 'transparent' }}
              >
                {activeCategory === category.id && (
                  <motion.span
                    layoutId="settings-category-switch-bubble"
                    className="absolute inset-0 z-0 bg-accent"
                    style={{ borderRadius: 6 }}
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <span className="relative z-10 flex items-center">
                  <category.icon size={16} className="mr-2 shrink-0" />
                  <span className="truncate">{category.label}</span>
                </span>
              </button>
            ))}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto overflow-x-hidden pr-2 -mr-2 custom-scrollbar">
          <AnimatePresence mode="wait">
            {activeCategory === 'general' && (
              <motion.div
                key="general"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-10"
              >
                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.general.title')}
                  </Text>
                  <div className="space-y-8">
                    <SettingItem label={t('settings.general.theme')} description={t('settings.general.themeDesc')}>
                      <Dropdown
                        onChange={(value: any) => onSettingsChange({ ...appSettings, theme: value })}
                        options={THEMES.map((theme: ThemeProps) => ({ value: theme.id, label: t(theme.name as any) }))}
                        value={appSettings?.theme || DEFAULT_THEME_ID}
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    <SettingItem label={t('settings.language')} description={t('settings.languageDesc')}>
                      <Dropdown
                        onChange={(value: any) => onSettingsChange({ ...appSettings, language: value })}
                        options={[
                          { value: 'en', label: 'English' },
                          { value: 'de', label: 'Deutsch' },
                        ]}
                        value={appSettings?.language || 'en'}
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    <div className="space-y-4">
                      <SettingItem
                        label={t('settings.general.xmpSync')}
                        description={t('settings.general.xmpSyncDesc')}
                      >
                        <Switch
                          checked={appSettings?.enableXmpSync ?? true}
                          id="enable-xmp-sync-toggle"
                          label={t('settings.general.enableXmpSync')}
                          onChange={(checked) => {
                            const newSettings = { ...appSettings, enableXmpSync: checked };
                            if (!checked) {
                              newSettings.createXmpIfMissing = false;
                            }
                            onSettingsChange(newSettings);
                          }}
                        />
                      </SettingItem>

                      <AnimatePresence initial={false}>
                        {(appSettings?.enableXmpSync ?? true) && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: 'easeInOut' }}
                            className="overflow-hidden"
                          >
                            <div className="pl-4 border-l-2 border-border-color ml-1">
                              <SettingItem
                                label={t('settings.general.createXmp')}
                                description={t('settings.general.createXmpDesc')}
                              >
                                <Switch
                                  checked={appSettings?.createXmpIfMissing ?? false}
                                  id="create-xmp-missing-toggle"
                                  label={t('settings.general.createXmpMissing')}
                                  onChange={(checked) =>
                                    onSettingsChange({ ...appSettings, createXmpIfMissing: checked })
                                  }
                                />
                              </SettingItem>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <SettingItem
                      label={t('settings.general.folderImageCounts')}
                      description={t('settings.general.folderImageCountsDesc')}
                    >
                      <Switch
                        checked={appSettings?.enableFolderImageCounts ?? false}
                        id="folder-image-counts-toggle"
                        label={t('settings.general.showImageCounts')}
                        onChange={(checked) => onSettingsChange({ ...appSettings, enableFolderImageCounts: checked })}
                      />
                    </SettingItem>

                    <SettingItem
                      label="Recently Opened Folders"
                      description="Keep the last 10 opened folders in the folder sidebar."
                    >
                      <Switch
                        checked={appSettings?.showRecentFolders ?? true}
                        id="show-recent-folders-toggle"
                        label="Show Recent Folders"
                        onChange={(checked) => onSettingsChange({ ...appSettings, showRecentFolders: checked })}
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.general.displayEditIcon')}
                      description={t('settings.general.displayEditIconDesc')}
                    >
                      <Switch
                        checked={appSettings?.displayEditIcon ?? true}
                        id="display-edit-icon-toggle"
                        label={t('settings.general.displayEditIcon')}
                        onChange={(checked) => onSettingsChange({ ...appSettings, displayEditIcon: checked })}
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.general.focusMode')}
                      description={t('settings.general.focusModeDesc')}
                    >
                      <Switch
                        checked={appSettings?.enableFocusMode ?? false}
                        id="focus-mode-toggle"
                        label={t('settings.general.enableFocusMode')}
                        onChange={(checked) => onSettingsChange({ ...appSettings, enableFocusMode: checked })}
                      />
                    </SettingItem>

                    <SettingItem label={t('settings.general.font')} description={t('settings.general.fontDesc')}>
                      <Dropdown
                        onChange={(value: any) => onSettingsChange({ ...appSettings, fontFamily: value })}
                        options={fontOptions}
                        value={appSettings?.fontFamily || 'poppins'}
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    {osPlatform === 'linux' && (
                      <SettingItem
                        label={t('settings.general.nativeTitlebar')}
                        description={t('settings.general.nativeTitlebarDesc')}
                      >
                        <Switch
                          checked={appSettings?.decorations ?? false}
                          id="native-titlebar-toggle"
                          label={t('settings.general.enableOsTitlebar')}
                          onChange={(checked) => {
                            onSettingsChange({ ...appSettings, decorations: checked });
                            getCurrentWindow().setDecorations(checked).catch(console.error);
                          }}
                        />
                      </SettingItem>
                    )}
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.adjustments.title')}
                  </Text>
                  <Text className="mb-4">{t('settings.adjustments.description')}</Text>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
                    <Switch
                      label={t('settings.adjustments.chromaticAberration')}
                      checked={appSettings?.adjustmentVisibility?.chromaticAberration ?? false}
                      onChange={(checked) =>
                        onSettingsChange({
                          ...appSettings,
                          adjustmentVisibility: {
                            ...(appSettings?.adjustmentVisibility || adjustmentVisibilityDefaults),
                            chromaticAberration: checked,
                          },
                        })
                      }
                    />
                    <Switch
                      label={t('settings.adjustments.grain')}
                      checked={appSettings?.adjustmentVisibility?.grain ?? true}
                      onChange={(checked) =>
                        onSettingsChange({
                          ...appSettings,
                          adjustmentVisibility: {
                            ...(appSettings?.adjustmentVisibility || adjustmentVisibilityDefaults),
                            grain: checked,
                          },
                        })
                      }
                    />
                    <Switch
                      label={t('settings.adjustments.colorCalibration')}
                      checked={appSettings?.adjustmentVisibility?.colorCalibration ?? true}
                      onChange={(checked) =>
                        onSettingsChange({
                          ...appSettings,
                          adjustmentVisibility: {
                            ...(appSettings?.adjustmentVisibility || adjustmentVisibilityDefaults),
                            colorCalibration: checked,
                          },
                        })
                      }
                    />
                    <Switch
                      label={t('settings.adjustments.noiseReduction')}
                      checked={appSettings?.adjustmentVisibility?.noiseReduction ?? true}
                      onChange={(checked) =>
                        onSettingsChange({
                          ...appSettings,
                          adjustmentVisibility: {
                            ...(appSettings?.adjustmentVisibility || adjustmentVisibilityDefaults),
                            noiseReduction: checked,
                          },
                        })
                      }
                    />
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.lenses.title')}
                  </Text>
                  <Text className="mb-6">{t('settings.lenses.description')}</Text>

                  <div className="space-y-8">
                    <div className="bg-bg-primary rounded-lg p-4 border border-border-color">
                      <Text variant={TextVariants.heading} className="mb-3">
                        {t('settings.lenses.addNew')}
                      </Text>
                      <div className="space-y-4">
                        <Dropdown
                          options={lensMakers.map((m) => ({ label: m, value: m }))}
                          value={tempLensMaker}
                          onChange={handleTempMakerChange}
                          placeholder={t('settings.lenses.manufacturerPlaceholder')}
                        />
                        <Dropdown
                          options={lensModels.map((m) => ({ label: m, value: m }))}
                          value={tempLensModel}
                          onChange={setTempLensModel}
                          placeholder={t('settings.lenses.modelPlaceholder')}
                          disabled={!tempLensMaker}
                        />
                        <Button onClick={handleAddLens} disabled={!tempLensMaker || !tempLensModel} className="w-full">
                          <Plus size={16} className="mr-1" />
                          {t('settings.lenses.addButton')}
                        </Button>
                      </div>
                    </div>

                    <div>
                      <Text variant={TextVariants.heading} className="mb-2">
                        {t('settings.lenses.saved')}
                      </Text>
                      {(!appSettings?.myLenses || appSettings.myLenses.length === 0) && (
                        <Text className="italic">{t('settings.lenses.noLenses')}</Text>
                      )}
                      <div className="divide-y divide-border-color">
                        {(appSettings?.myLenses || []).map((lens: MyLens, index: number) => (
                          <div
                            key={`${lens.maker}-${lens.model}-${index}`}
                            className="flex justify-between items-center py-3 first:pt-0 last:pb-0"
                          >
                            <div className="flex items-center gap-3">
                              <div className="p-2 bg-surface rounded-md text-accent">
                                <Bookmark size={16} />
                              </div>
                              <div>
                                <Text color={TextColors.primary} weight={TextWeights.medium}>
                                  {lens.model}
                                </Text>
                                <Text variant={TextVariants.small}>{lens.maker}</Text>
                              </div>
                            </div>
                            <button
                              onClick={() => handleRemoveLens(index)}
                              className="p-2 text-text-secondary hover:text-red-400 hover:bg-bg-primary rounded-md transition-colors"
                              data-tooltip={t('settings.lenses.removeTooltip')}
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.tagging.title')}
                  </Text>
                  <div className="space-y-8">
                    <div className="space-y-4">
                      <SettingItem
                        description={t('settings.tagging.aiTaggingDesc')}
                        label={t('settings.tagging.aiTagging')}
                      >
                        <Switch
                          checked={appSettings?.enableAiTagging ?? false}
                          id="ai-tagging-toggle"
                          label={t('settings.tagging.automaticAiTagging')}
                          onChange={(checked) => onSettingsChange({ ...appSettings, enableAiTagging: checked })}
                        />
                      </SettingItem>

                      <AnimatePresence>
                        {(appSettings?.enableAiTagging ?? false) && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: 'easeInOut' }}
                            className="overflow-hidden"
                          >
                            <div className="pl-4 border-l-2 border-border-color ml-1 space-y-8">
                              <SettingItem
                                label={t('settings.tagging.maxAiTags')}
                                description={t('settings.tagging.maxAiTagsDesc')}
                              >
                                <Slider
                                  label={t('settings.tagging.amount')}
                                  min={1}
                                  max={20}
                                  step={1}
                                  value={appSettings?.aiTagCount ?? 10}
                                  defaultValue={10}
                                  onChange={(e: any) =>
                                    onSettingsChange({ ...appSettings, aiTagCount: parseInt(e.target.value) })
                                  }
                                />
                              </SettingItem>

                              <SettingItem
                                label={t('settings.tagging.customList')}
                                description={t('settings.tagging.customListDesc')}
                              >
                                <div>
                                  <div className="flex flex-wrap gap-2 p-2 bg-bg-primary rounded-md min-h-10 border border-border-color mb-2 items-center">
                                    <AnimatePresence>
                                      {customAiTags.length > 0 ? (
                                        customAiTags.map((tag: string) => (
                                          <motion.div
                                            key={tag}
                                            layout
                                            variants={shortcutTagVariants}
                                            initial={false}
                                            animate="visible"
                                            exit="exit"
                                            onClick={() => handleRemoveAiTag(tag)}
                                            data-tooltip={t('settings.tagging.removeCustomTooltip', { tag })}
                                            className="flex items-center gap-1 bg-surface px-2 py-1 rounded-sm group cursor-pointer"
                                          >
                                            <Text variant={TextVariants.label} color={TextColors.primary}>
                                              {tag}
                                            </Text>
                                            <span className="rounded-full group-hover:bg-black/20 p-0.5 transition-colors">
                                              <X size={14} />
                                            </span>
                                          </motion.div>
                                        ))
                                      ) : (
                                        <motion.span
                                          key="no-ai-tags-placeholder"
                                          initial={{ opacity: 0 }}
                                          animate={{ opacity: 1 }}
                                          exit={{ opacity: 0 }}
                                          transition={{ duration: 0.2 }}
                                        >
                                          <Text className="px-1 select-none italic">
                                            {t('settings.tagging.noCustomTags')}
                                          </Text>
                                        </motion.span>
                                      )}
                                    </AnimatePresence>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <div className="relative flex-1">
                                      <Input
                                        type="text"
                                        value={newAiTag}
                                        onChange={(e) => setNewAiTag(e.target.value)}
                                        onKeyDown={handleAiTagInputKeyDown}
                                        placeholder={t('settings.tagging.addCustomPlaceholder')}
                                        className="pr-10"
                                        bgClassName="bg-bg-primary"
                                      />
                                      <button
                                        onClick={handleAddAiTag}
                                        className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-text-secondary hover:text-text-primary hover:bg-surface"
                                        data-tooltip={t('settings.tagging.addCustomTooltip')}
                                      >
                                        <Plus size={18} />
                                      </button>
                                    </div>
                                    <button
                                      onClick={() => onSettingsChange({ ...appSettings, customAiTags: [] })}
                                      disabled={customAiTags.length === 0}
                                      className="p-2 text-text-secondary hover:text-red-400 hover:bg-surface rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:text-text-secondary disabled:hover:bg-transparent"
                                      data-tooltip={t('settings.tagging.clearCustomTooltip')}
                                    >
                                      <Trash2 size={18} />
                                    </button>
                                  </div>
                                </div>
                              </SettingItem>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <SettingItem
                      label={t('settings.tagging.shortcuts')}
                      description={t('settings.tagging.shortcutsDesc')}
                    >
                      <div>
                        <div className="flex flex-wrap gap-2 p-2 bg-bg-primary rounded-md min-h-10 border border-border-color mb-2 items-center">
                          <AnimatePresence>
                            {taggingShortcuts.length > 0 ? (
                              taggingShortcuts.map((shortcut: string) => (
                                <motion.div
                                  key={shortcut}
                                  layout
                                  variants={shortcutTagVariants}
                                  initial={false}
                                  animate="visible"
                                  exit="exit"
                                  onClick={() => handleRemoveShortcut(shortcut)}
                                  data-tooltip={t('settings.tagging.removeShortcutTooltip', { shortcut })}
                                  className="flex items-center gap-1 bg-surface px-2 py-1 rounded-sm group cursor-pointer"
                                >
                                  <Text variant={TextVariants.label} color={TextColors.primary}>
                                    {shortcut}
                                  </Text>
                                  <span className="rounded-full group-hover:bg-black/20 p-0.5 transition-colors">
                                    <X size={14} />
                                  </span>
                                </motion.div>
                              ))
                            ) : (
                              <motion.span
                                key="no-shortcuts-placeholder"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="text-sm text-text-secondary italic px-1 select-none"
                              >
                                {t('settings.tagging.noShortcuts')}
                              </motion.span>
                            )}
                          </AnimatePresence>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="relative flex-1">
                            <Input
                              type="text"
                              value={newShortcut}
                              onChange={(e) => setNewShortcut(e.target.value)}
                              onKeyDown={handleInputKeyDown}
                              placeholder={t('settings.tagging.addShortcutsPlaceholder')}
                              className="pr-10"
                              bgClassName="bg-bg-primary"
                            />
                            <button
                              onClick={handleAddShortcut}
                              className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-text-secondary hover:text-text-primary hover:bg-surface"
                              data-tooltip={t('settings.tagging.addShortcutTooltip')}
                            >
                              <Plus size={18} />
                            </button>
                          </div>
                          <button
                            onClick={() => onSettingsChange({ ...appSettings, taggingShortcuts: [] })}
                            disabled={taggingShortcuts.length === 0}
                            className="p-2 text-text-secondary hover:text-red-400 hover:bg-surface rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:text-text-secondary disabled:hover:bg-transparent"
                            data-tooltip={t('settings.tagging.clearShortcutsTooltip')}
                          >
                            <Trash2 size={18} />
                          </button>
                        </div>
                      </div>
                    </SettingItem>

                    <div className="pt-8 border-t border-border-color">
                      <div className="space-y-8">
                        <DataActionItem
                          buttonAction={handleClearAiTags}
                          buttonText={t('settings.tagging.clearAiTagsButton')}
                          description={t('settings.tagging.clearAiTagsDesc')}
                          disabled={effectiveRootPaths.length === 0}
                          icon={<Trash2 size={16} className="mr-2" />}
                          isProcessing={isClearingAiTags}
                          message={aiTagsClearMessage}
                          title={t('settings.tagging.clearAiTagsTitle')}
                        />
                        <DataActionItem
                          buttonAction={handleClearTags}
                          buttonText={t('settings.tagging.clearAiTagsButton')}
                          description={t('settings.tagging.clearAllTagsDesc')}
                          disabled={effectiveRootPaths.length === 0}
                          icon={<Trash2 size={16} className="mr-2" />}
                          isProcessing={isClearingTags}
                          message={tagsClearMessage}
                          title={t('settings.tagging.clearAllTagsTitle')}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-6">
                    {t('settings.thanks.title')}
                  </Text>
                  <Text className="mb-4">{t('settings.thanks.description')}</Text>
                  <Text as="ul" className="space-y-3 list-disc ml-5 pl-1">
                    <li>
                      <a
                        href="https://github.com/dnglab/dnglab/tree/main/rawler"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        rawler
                      </a>
                      : {t('settings.thanks.list.rawler')}
                    </li>
                    <li>
                      <a
                        href="https://lensfun.github.io/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        lensfun
                      </a>
                      : {t('settings.thanks.list.lensfun')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/marcinz606/NegPy"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        NegPy
                      </a>
                      : {t('settings.thanks.list.negpy')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/advimman/lama"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        LaMa
                      </a>
                      : {t('settings.thanks.list.lama')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/facebookresearch/sam2"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        SAM 2
                      </a>
                      : {t('settings.thanks.list.sam2')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/xuebinqin/U-2-Net"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        U-2-Net
                      </a>
                      : {t('settings.thanks.list.u2net')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/DepthAnything/Depth-Anything-V2"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        Depth Anything V2
                      </a>
                      : {t('settings.thanks.list.depth')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/trougnouf/nind-denoise"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        nind-denoise
                      </a>
                      : {t('settings.thanks.list.nind')}
                    </li>
                    <li>
                      <a
                        href="https://github.com/darktable-org/darktable"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-accent hover:underline"
                      >
                        darktable & co.
                      </a>
                      : {t('settings.thanks.list.darktable')}
                    </li>
                    <li>
                      <span className="font-semibold text-accent">{t('settings.thanks.list.youLabel')}</span>:{' '}
                      {t('settings.thanks.list.you')}
                    </li>
                  </Text>
                </div>
              </motion.div>
            )}
            {activeCategory === 'processing' && (
              <motion.div
                key="processing"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-10"
              >
                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.processing.title')}
                  </Text>
                  <div className="space-y-8">
                    <div>
                      <Text variant={TextVariants.heading} className="mb-2">
                        {t('settings.processing.previewStrategy')}
                      </Text>
                      <PreviewModeSwitch
                        mode={appSettings?.enableZoomHifi ? 'dynamic' : 'static'}
                        onModeChange={handlePreviewModeChange}
                      />

                      <div className="mt-3">
                        <AnimatePresence mode="wait">
                          {!(appSettings?.enableZoomHifi ?? true) ? (
                            <motion.div
                              key="static-preview"
                              initial={{ opacity: 0, x: 10 }}
                              animate={{ opacity: 1, x: 0 }}
                              exit={{ opacity: 0, x: -10 }}
                              transition={{ duration: 0.2 }}
                            >
                              <Text variant={TextVariants.small} className="mb-4">
                                {t('settings.processing.staticDesc')}
                              </Text>
                              <div className="pl-4 border-l-2 border-border-color ml-1">
                                <SettingItem
                                  description={t('settings.processing.previewResDesc')}
                                  label={t('settings.processing.previewRes')}
                                >
                                  <Dropdown
                                    onChange={(value: any) =>
                                      handleProcessingSettingChange('editorPreviewResolution', value)
                                    }
                                    options={resolutions}
                                    value={processingSettings.editorPreviewResolution}
                                    triggerClassName="bg-bg-primary"
                                  />
                                </SettingItem>
                              </div>
                            </motion.div>
                          ) : (
                            <motion.div
                              key="dynamic-preview"
                              initial={{ opacity: 0, x: 10 }}
                              animate={{ opacity: 1, x: 0 }}
                              exit={{ opacity: 0, x: -10 }}
                              transition={{ duration: 0.2 }}
                            >
                              <Text variant={TextVariants.small} className="mb-4">
                                {t('settings.processing.dynamicDesc')}
                              </Text>
                              <div className="pl-4 border-l-2 border-border-color ml-1 space-y-3">
                                <SettingItem
                                  description={t('settings.processing.staticPreviewResDesc')}
                                  label={t('settings.processing.staticPreviewRes')}
                                >
                                  <Dropdown
                                    onChange={(value: any) =>
                                      handleProcessingSettingChange('editorPreviewResolution', value)
                                    }
                                    options={resolutions}
                                    value={processingSettings.editorPreviewResolution}
                                    triggerClassName="bg-bg-primary"
                                  />
                                </SettingItem>

                                <SettingItem
                                  label={t('settings.processing.renderScale')}
                                  description={t('settings.processing.renderScaleDesc')}
                                >
                                  <Dropdown
                                    onChange={(value: any) =>
                                      handleProcessingSettingChange('highResZoomMultiplier', value)
                                    }
                                    options={zoomMultiplierOptions}
                                    value={processingSettings.highResZoomMultiplier}
                                    triggerClassName="bg-bg-primary"
                                  />
                                </SettingItem>

                                <SettingItem
                                  label={t('settings.processing.highDpi')}
                                  description={
                                    dpr > 1
                                      ? t('settings.processing.highDpiDesc', { dpr })
                                      : t('settings.processing.highDpiDescStandard')
                                  }
                                >
                                  <Switch
                                    checked={processingSettings.useFullDpiRendering}
                                    disabled={dpr <= 1}
                                    id="full-dpi-rendering-toggle"
                                    label={t('settings.processing.nativeDpi')}
                                    onChange={(checked) =>
                                      handleProcessingSettingChange('useFullDpiRendering', checked)
                                    }
                                  />
                                </SettingItem>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <SettingItem
                        label={t('settings.processing.livePreviews')}
                        description={t('settings.processing.livePreviewsDesc')}
                      >
                        <Switch
                          checked={appSettings?.enableLivePreviews ?? true}
                          id="live-previews-toggle"
                          label={t('settings.processing.enableLivePreviews')}
                          onChange={(checked) => {
                            setHasInteractedWithLivePreview(true);
                            onSettingsChange({ ...appSettings, enableLivePreviews: checked });
                          }}
                        />
                      </SettingItem>

                      <AnimatePresence>
                        {(appSettings?.enableLivePreviews ?? true) && (
                          <motion.div
                            initial={hasInteractedWithLivePreview ? { height: 0, opacity: 0 } : false}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: 'easeInOut' }}
                          >
                            <div className="pl-4 border-l-2 border-border-color ml-1">
                              <SettingItem
                                label={t('settings.processing.livePreviewQuality')}
                                description={t('settings.processing.livePreviewQualityDesc')}
                              >
                                <Dropdown
                                  onChange={(value: any) =>
                                    onSettingsChange({ ...appSettings, livePreviewQuality: value })
                                  }
                                  options={livePreviewQualityOptions}
                                  value={appSettings?.livePreviewQuality || 'high'}
                                  triggerClassName="bg-bg-primary"
                                />
                              </SettingItem>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <SettingItem
                      description={t('settings.processing.thumbnailResDesc')}
                      label={t('settings.processing.thumbnailRes')}
                    >
                      <Dropdown
                        onChange={(value: any) => handleProcessingSettingChange('thumbnailResolution', value)}
                        options={thumbnailResolutions}
                        value={processingSettings.thumbnailResolution}
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.workerThreads')}
                      description={t('settings.processing.workerThreadsDesc')}
                    >
                      <Slider
                        label={t('settings.processing.threads')}
                        min={2}
                        max={10}
                        step={1}
                        value={processingSettings.thumbnailWorkerThreads}
                        defaultValue={4}
                        onChange={(e: any) =>
                          handleProcessingSettingChange('thumbnailWorkerThreads', parseInt(e.target.value))
                        }
                        fillOrigin="min"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.imageCache')}
                      description={t('settings.processing.imageCacheDesc')}
                    >
                      <Slider
                        label={t('settings.processing.images')}
                        min={2}
                        max={10}
                        step={1}
                        value={processingSettings.imageCacheSize}
                        defaultValue={5}
                        onChange={(e: any) => handleProcessingSettingChange('imageCacheSize', parseInt(e.target.value))}
                        fillOrigin="min"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.wgpu')}
                      description={
                        osPlatform === 'linux'
                          ? t('settings.processing.wgpuDescLinux')
                          : osPlatform === 'android'
                            ? t('settings.processing.wgpuDescAndroid')
                            : t('settings.processing.wgpuDescRecommended')
                      }
                    >
                      <Switch
                        checked={processingSettings.useWgpuRenderer}
                        disabled={osPlatform === 'linux' || osPlatform === 'android'}
                        id="wgpu-renderer-toggle"
                        label={t('settings.processing.wgpuLabel')}
                        onChange={(checked) => handleProcessingSettingChange('useWgpuRenderer', checked)}
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.backend')}
                      description={t('settings.processing.backendDesc')}
                    >
                      <Dropdown
                        onChange={(value: any) => handleProcessingSettingChange('processingBackend', value)}
                        options={filteredBackendOptions}
                        value={
                          filteredBackendOptions.some((option) => option.value === processingSettings.processingBackend)
                            ? processingSettings.processingBackend
                            : 'auto'
                        }
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    {osPlatform !== 'macos' && osPlatform !== 'windows' && (
                      <SettingItem
                        label={t('settings.processing.linuxCompat')}
                        description={t('settings.processing.linuxCompatDesc')}
                      >
                        <Switch
                          checked={processingSettings.linuxGpuOptimization}
                          id="gpu-compat-toggle"
                          label={t('settings.processing.linuxCompatLabel')}
                          onChange={(checked) => handleProcessingSettingChange('linuxGpuOptimization', checked)}
                        />
                      </SettingItem>
                    )}

                    {restartRequired && (
                      <>
                        <Text
                          as="div"
                          color={TextColors.info}
                          className="p-3 bg-blue-900/10 border border-blue-500/50 rounded-lg flex items-center gap-3"
                        >
                          <Info size={18} />
                          <p>{t('settings.processing.restartRequired')}</p>
                        </Text>
                        <div className="flex justify-end">
                          <Button onClick={handleSaveAndRelaunch}>{t('settings.processing.saveRelaunch')}</Button>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.processing.preprocessing.title')}
                  </Text>
                  <div className="space-y-8">
                    <SettingItem
                      label={t('settings.processing.preprocessing.highlightRecovery')}
                      description={t('settings.processing.preprocessing.highlightRecoveryDesc')}
                    >
                      <Slider
                        label={t('settings.tagging.amount')}
                        min={1}
                        max={10}
                        step={0.1}
                        value={processingSettings.rawHighlightCompression}
                        defaultValue={2.5}
                        onChange={(e: any) =>
                          handleProcessingSettingChange('rawHighlightCompression', parseFloat(e.target.value))
                        }
                        fillOrigin="min"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.preprocessing.colorNr')}
                      description={t('settings.processing.preprocessing.colorNrDesc')}
                    >
                      <Slider
                        label={t('settings.tagging.amount')}
                        min={0}
                        max={1.0}
                        step={0.05}
                        value={processingSettings.rawPreprocessingColorNr}
                        defaultValue={0.5}
                        onChange={(e: any) =>
                          handleProcessingSettingChange('rawPreprocessingColorNr', parseFloat(e.target.value))
                        }
                        fillOrigin="min"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.preprocessing.sharpening')}
                      description={t('settings.processing.preprocessing.sharpeningDesc')}
                    >
                      <Slider
                        label={t('settings.tagging.amount')}
                        min={0}
                        max={1.0}
                        step={0.05}
                        value={processingSettings.rawPreprocessingSharpening}
                        defaultValue={0.35}
                        onChange={(e: any) =>
                          handleProcessingSettingChange('rawPreprocessingSharpening', parseFloat(e.target.value))
                        }
                        fillOrigin="min"
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.preprocessing.applyPreprocessing')}
                      description={t('settings.processing.preprocessing.applyPreprocessingDesc')}
                    >
                      <Switch
                        checked={processingSettings.applyPreprocessingToNonRaws}
                        id="preprocessing-non-raws-toggle"
                        label={t('settings.processing.preprocessing.enablePreprocessingNonRaws')}
                        onChange={(checked) => handleProcessingSettingChange('applyPreprocessingToNonRaws', checked)}
                      />
                    </SettingItem>

                    <SettingItem
                      label={t('settings.processing.preprocessing.linearRaw')}
                      description={t('settings.processing.preprocessing.linearRawDesc')}
                    >
                      <Dropdown
                        onChange={(value: any) => onSettingsChange({ ...appSettings, linearRawMode: value })}
                        options={linearRawOptions}
                        value={appSettings?.linearRawMode || 'auto'}
                        triggerClassName="bg-bg-primary"
                      />
                    </SettingItem>

                    <div className="space-y-4">
                      <SettingItem
                        label={t('settings.processing.preprocessing.tonemapperOverride')}
                        description={t('settings.processing.preprocessing.tonemapperOverrideDesc')}
                      >
                        <Switch
                          checked={appSettings?.tonemapperOverrideEnabled ?? false}
                          id="tonemapper-override-toggle"
                          label={t('settings.processing.preprocessing.enableTonemapperOverride')}
                          onChange={(checked) =>
                            onSettingsChange({ ...appSettings, tonemapperOverrideEnabled: checked })
                          }
                        />
                      </SettingItem>

                      <AnimatePresence>
                        {(appSettings?.tonemapperOverrideEnabled ?? false) && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: 'easeInOut' }}
                          >
                            <div className="pl-4 border-l-2 border-border-color ml-1 space-y-3">
                              <SettingItem
                                label={t('settings.processing.preprocessing.defaultRawTonemapper')}
                                description={t('settings.processing.preprocessing.defaultRawTonemapperDesc')}
                              >
                                <Dropdown
                                  onChange={(value: any) =>
                                    onSettingsChange({ ...appSettings, defaultRawTonemapper: value })
                                  }
                                  options={tonemapperOptions}
                                  value={appSettings?.defaultRawTonemapper || 'agx'}
                                  triggerClassName="bg-bg-primary"
                                />
                              </SettingItem>

                              <SettingItem
                                label={t('settings.processing.preprocessing.defaultNonRawTonemapper')}
                                description={t('settings.processing.preprocessing.defaultNonRawTonemapperDesc')}
                              >
                                <Dropdown
                                  onChange={(value: any) =>
                                    onSettingsChange({ ...appSettings, defaultNonRawTonemapper: value })
                                  }
                                  options={tonemapperOptions}
                                  value={appSettings?.defaultNonRawTonemapper || 'basic'}
                                  triggerClassName="bg-bg-primary"
                                />
                              </SettingItem>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.processing.ai.title')}
                  </Text>
                  <Text className="mb-4">{t('settings.processing.ai.description')}</Text>

                  <AiProviderSwitch selectedProvider={aiProvider} onProviderChange={handleProviderChange} />

                  <div className="mt-8">
                    <AnimatePresence mode="wait">
                      {aiProvider === 'cpu' && (
                        <motion.div
                          key="cpu"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                        >
                          <Text variant={TextVariants.heading}>{t('settings.processing.ai.cpu.title')}</Text>
                          <Text className="mt-1">{t('settings.processing.ai.cpu.description')}</Text>
                          <Text as="ul" className="mt-3 space-y-1 list-disc list-inside">
                            <li>{t('settings.processing.ai.cpu.feature1')}</li>
                            <li>{t('settings.processing.ai.cpu.feature2')}</li>
                            <li>{t('settings.processing.ai.cpu.feature3')}</li>
                          </Text>
                        </motion.div>
                      )}

                      {aiProvider === 'local-gpu' && (
                        <motion.div
                          key="local-gpu"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                        >
                          <div className="space-y-6">
                            <div>
                              <Text variant={TextVariants.heading}>Local GPU</Text>
                              <Text className="mt-1">
                                Runs local CUDA inpainting and a managed SDXL workflow from RapidRAW's install folder.
                                Use LaMa for quick cleanup or SDXL for prompt-based generative replace.
                              </Text>
                            </div>

                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                              <div className="p-4 bg-bg-primary rounded-lg border border-border-color">
                                <div className="flex items-center gap-2 mb-2">
                                  <Zap size={16} />
                                  <Text weight={TextWeights.semibold}>GPU</Text>
                                </div>
                                <Text variant={TextVariants.small} className="block">
                                  {localAiStatus?.gpu.name || 'No NVIDIA GPU detected'}
                                </Text>
                                <Text variant={TextVariants.small} className="block mt-1">
                                  Driver: {localAiStatus?.gpu.driverVersion || 'Unknown'}
                                </Text>
                                <Text variant={TextVariants.small} className="block mt-1">
                                  VRAM: {localAiStatus?.gpu.vramMb ? `${localAiStatus.gpu.vramMb} MB` : 'Unknown'}
                                </Text>
                                <Text variant={TextVariants.small} className="block mt-1">
                                  Compute: {localAiStatus?.gpu.computeCapability || 'Unknown'}
                                </Text>
                              </div>

                              <div className="p-4 bg-bg-primary rounded-lg border border-border-color">
                                <div className="flex items-center gap-2 mb-2">
                                  <HardDrive size={16} />
                                  <Text weight={TextWeights.semibold}>Model Storage</Text>
                                </div>
                                <Text variant={TextVariants.small} className="block break-all">
                                  {localAiStatus?.modelDir || 'Loading...'}
                                </Text>
                                <Text variant={TextVariants.small} className="block mt-2">
                                  Managed file types: {localAiStatus?.requiredFileTypes.join(', ') || '.onnx, .safetensors'}
                                </Text>
                                <Text variant={TextVariants.small} className="block mt-1">
                                  Disk usage: {formatBytes(localAiStatus?.diskUsageBytes || 0)}
                                </Text>
                              </div>
                            </div>

                            <div className="p-4 bg-bg-primary rounded-lg border border-border-color space-y-4">
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                  <Text weight={TextWeights.semibold}>CUDA Runtime</Text>
                                  <Text variant={TextVariants.small} className="block mt-1">
                                    {runtimeStatusLabel}
                                  </Text>
                                </div>
                                <Button
                                  className={secondaryLocalAiButtonClass}
                                  disabled={isLocalAiBusy}
                                  onClick={() => refreshLocalAiStatus(true, 'runtime-refresh')}
                                >
                                  <RefreshCw
                                    size={16}
                                    className={localAiTask === 'runtime-refresh' ? 'animate-spin' : ''}
                                  />
                                  {localAiTask === 'runtime-refresh' ? 'Checking...' : 'Refresh'}
                                </Button>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                <Text
                                  as="div"
                                  color={cudaRuntimeReady ? TextColors.success : TextColors.error}
                                  variant={TextVariants.small}
                                  className="flex items-center gap-2 rounded-md bg-surface px-3 py-2"
                                >
                                  {cudaRuntimeReady ? <Wifi size={14} /> : <WifiOff size={14} />}
                                  CUDA Toolkit {cudaRuntimeReady ? 'found' : 'not found'}
                                </Text>
                                <Text
                                  as="div"
                                  color={cudnnRuntimeReady ? TextColors.success : TextColors.error}
                                  variant={TextVariants.small}
                                  className="flex items-center gap-2 rounded-md bg-surface px-3 py-2"
                                >
                                  {cudnnRuntimeReady ? <Wifi size={14} /> : <WifiOff size={14} />}
                                  cuDNN 9 {cudnnRuntimeReady ? 'found' : 'not found'}
                                </Text>
                              </div>

                              {!!missingLocalAiRuntimeDependencies.length && (
                                <div className="space-y-2 rounded-md bg-surface p-3">
                                  <Text weight={TextWeights.semibold}>Install missing runtime</Text>
                                  {missingCudaRuntime && (
                                    <Text variant={TextVariants.small} className="block">
                                      <a
                                        className="font-semibold text-accent hover:underline"
                                        href={CUDA_DOWNLOAD_URL}
                                        rel="noopener noreferrer"
                                        target="_blank"
                                      >
                                        Download CUDA Toolkit 12.x
                                      </a>
                                      , install it for Windows, then refresh.
                                    </Text>
                                  )}
                                  {missingCudnnRuntime && (
                                    <Text variant={TextVariants.small} className="block">
                                      <a
                                        className="font-semibold text-accent hover:underline"
                                        href={CUDNN_DOWNLOAD_URL}
                                        rel="noopener noreferrer"
                                        target="_blank"
                                      >
                                        Download cuDNN 9
                                      </a>
                                      , install it for Windows, then refresh. Use NVIDIA’s{' '}
                                      <a
                                        className="font-semibold text-accent hover:underline"
                                        href={CUDNN_WINDOWS_INSTALL_GUIDE_URL}
                                        rel="noopener noreferrer"
                                        target="_blank"
                                      >
                                        Windows guide
                                      </a>
                                      {' '}if needed.
                                    </Text>
                                  )}
                                </div>
                              )}

                              <details className="rounded-md bg-surface p-3">
                                <summary className="cursor-pointer text-sm font-semibold">Advanced details</summary>
                                <div className="mt-3 space-y-3">
                                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        CUDA bin folder
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        onBlur={handleLocalAiRuntimePathSave}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          setLocalAiCudaRuntimePath(e.target.value)
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        placeholder="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin"
                                        value={localAiCudaRuntimePath}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        cuDNN 9 bin folder
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        onBlur={handleLocalAiRuntimePathSave}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          setLocalAiCudnnRuntimePath(e.target.value)
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        placeholder="C:\Program Files\NVIDIA\CUDNN\v9.x\bin"
                                        value={localAiCudnnRuntimePath}
                                      />
                                    </label>
                                  </div>
                                  {!!missingLocalAiRuntimeDependencies.length && (
                                    <Text color={TextColors.error} variant={TextVariants.small} className="block">
                                      Missing files: {missingLocalAiRuntimeDependencies.join(', ')}
                                    </Text>
                                  )}
                                  {!!localAiRuntimeDependencies.length && (
                                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
                                      {localAiRuntimeDependencies.map((dependency) => (
                                        <Text
                                          key={`${dependency.kind}-${dependency.name}`}
                                          as="div"
                                          color={dependency.found ? TextColors.success : TextColors.error}
                                          variant={TextVariants.small}
                                          className="min-w-0 rounded-md bg-bg-primary px-3 py-2"
                                        >
                                          <span className="font-medium">
                                            {dependency.name} ({dependency.kind})
                                          </span>
                                          <span className="block break-all text-text-secondary">
                                            {dependency.path || 'Missing'}
                                          </span>
                                        </Text>
                                      ))}
                                    </div>
                                  )}
                                  <Button
                                    className={secondaryLocalAiButtonClass}
                                    disabled={isLocalAiBusy}
                                    onClick={handleLocalAiRuntimePathSave}
                                  >
                                    <RefreshCw
                                      size={16}
                                      className={localAiTask === 'save-runtime' ? 'animate-spin' : ''}
                                    />
                                    {localAiTask === 'save-runtime' ? 'Saving...' : 'Save Paths'}
                                  </Button>
                                </div>
                              </details>
                            </div>

                            <div className="p-4 bg-bg-primary rounded-lg border border-border-color space-y-4">
                              <div className="flex flex-wrap items-start justify-between gap-3">
                                <div>
                                  <Text weight={TextWeights.semibold}>Generative SDXL</Text>
                                  <Text variant={TextVariants.small} className="block mt-1">
                                    Prompt-based masked generation using the same SDXL, ControlNet, and VAE model set
                                    as AI Connector.
                                  </Text>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full xl:w-auto">
                                  <div className="space-y-2">
                                    <Text variant={TextVariants.small} weight={TextWeights.semibold} className="block">
                                      Setup
                                    </Text>
                                    <div className="space-y-2">
                                      <div className="flex items-center gap-2">
                                        <Button
                                          className={localAiGroupedButtonClass}
                                          disabled={isLocalAiBusy || !!localAiStatus?.localComfy?.runtimeInstalled}
                                          onClick={handleDownloadLocalAiRuntime}
                                        >
                                          <Download
                                            size={16}
                                            className={localAiTask === 'runtime-download' ? 'animate-pulse' : ''}
                                          />
                                          {localAiTask === 'runtime-download' ? 'Installing...' : 'Install Runtime'}
                                        </Button>
                                        <Button
                                          className={localAiIconButtonClass}
                                          data-tooltip="Delete runtime"
                                          disabled={isLocalAiBusy || !localAiStatus?.localComfy?.runtimeInstalled}
                                          onClick={handleDeleteLocalAiRuntime}
                                          title="Delete runtime"
                                        >
                                          <Trash2
                                            size={16}
                                            className={clsx(
                                              '!h-4 !w-4 shrink-0',
                                              localAiTask === 'runtime-delete' && 'animate-pulse',
                                            )}
                                          />
                                        </Button>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <Button
                                          className={localAiGroupedButtonClass}
                                          disabled={
                                            isLocalAiBusy ||
                                            !localAiStatus?.modelDirWritable ||
                                            localAiGenerativeModelsReady
                                          }
                                          onClick={handleDownloadLocalAiGenerativeAssets}
                                        >
                                          <Download
                                            size={16}
                                            className={
                                              localAiTask === 'generative-download' && !localAiDownloadProgress
                                                ? 'animate-pulse'
                                                : ''
                                            }
                                          />
                                          {localAiTask === 'generative-download' ? 'Downloading...' : 'Download Models'}
                                        </Button>
                                        <Button
                                          className={localAiIconButtonClass}
                                          data-tooltip="Delete models"
                                          disabled={isLocalAiBusy || !localAiGenerativeModels.some((model) => model.installed)}
                                          onClick={handleDeleteLocalAiGenerativeAssets}
                                          title="Delete models"
                                        >
                                          <Trash2
                                            size={16}
                                            className={clsx(
                                              '!h-4 !w-4 shrink-0',
                                              localAiTask === 'generative-delete' && 'animate-pulse',
                                            )}
                                          />
                                        </Button>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="space-y-2 md:pl-4">
                                    <Text variant={TextVariants.small} weight={TextWeights.semibold} className="block">
                                      Verify
                                    </Text>
                                    <div className="flex items-start gap-2">
                                      <Button
                                        className="w-full sm:w-40 justify-start whitespace-nowrap"
                                        disabled={isLocalAiBusy || !localAiGenerativeReady}
                                        onClick={handleRunLocalGenerativeSelfTest}
                                      >
                                        <PlayCircle
                                          size={16}
                                          className={localAiTask === 'generative-test' ? 'animate-pulse' : ''}
                                        />
                                        {localAiTask === 'generative-test' ? 'Testing...' : 'Run Test'}
                                      </Button>
                                    </div>
                                  </div>

                                </div>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                                <Text
                                  as="div"
                                  color={localAiStatus?.localComfy?.runtimeInstalled ? TextColors.success : TextColors.info}
                                  variant={TextVariants.small}
                                  className="rounded-md bg-surface px-3 py-2"
                                >
                                  Runtime: {localAiStatus?.localComfy?.runtimeInstalled ? 'Installed' : 'Not installed'}
                                </Text>
                                <Text
                                  as="div"
                                  color={localAiGenerativeModelsReady ? TextColors.success : TextColors.info}
                                  variant={TextVariants.small}
                                  className="rounded-md bg-surface px-3 py-2"
                                >
                                  Models: {localAiGenerativeModelsReady ? 'Installed' : 'Missing'}
                                </Text>
                                <Text
                                  as="div"
                                  color={localAiStatus?.localComfy?.running ? TextColors.success : TextColors.info}
                                  variant={TextVariants.small}
                                  className="rounded-md bg-surface px-3 py-2"
                                >
                                  Service: {localAiStatus?.localComfy?.running ? `Running on ${localAiStatus.localComfy.port}` : 'Stopped'}
                                </Text>
                              </div>

                              <Text variant={TextVariants.small} className="block break-all">
                                Runtime folder: {localAiStatus?.localComfy?.runtimeDir || 'Loading...'}
                              </Text>
                              <Text variant={TextVariants.small} className="block">
                                SDXL model file type: .safetensors
                              </Text>

                              {localAiGenerativeTask && (
                                <div className="space-y-2" aria-live="polite">
                                  <div className="flex items-center gap-2">
                                    <RefreshCw size={14} className="animate-spin text-accent" />
                                    <Text variant={TextVariants.small} color={TextColors.accent}>
                                      {localAiTaskLabel}
                                    </Text>
                                  </div>
                                  <div className="h-2 overflow-hidden rounded-full bg-surface" role="progressbar">
                                    <div className="h-full w-1/3 animate-pulse rounded-full bg-accent" />
                                  </div>
                                </div>
                              )}

                              {localAiDownloadProgress &&
                                (localAiTask === 'generative-download' || localAiTask === 'runtime-download') && (
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between gap-3">
                                    <Text variant={TextVariants.small} color={TextColors.accent}>
                                      {localAiDownloadProgress.modelName}: {localAiDownloadLabel}
                                    </Text>
                                    {localAiDownloadPercent !== null && (
                                      <Text variant={TextVariants.small} color={TextColors.accent}>
                                        {localAiDownloadPercent}%
                                      </Text>
                                    )}
                                  </div>
                                  <div
                                    className="h-2 overflow-hidden rounded-full bg-surface"
                                    role="progressbar"
                                    aria-label="Generative SDXL model download progress"
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-valuenow={localAiDownloadPercent ?? undefined}
                                  >
                                    <div
                                      className={clsx('h-full rounded-full bg-accent transition-all', {
                                        'animate-pulse': localAiDownloadPercent === null,
                                      })}
                                      style={{
                                        width: localAiDownloadPercent === null ? '35%' : `${localAiDownloadPercent}%`,
                                      }}
                                    />
                                  </div>
                                </div>
                              )}

                              <details className="rounded-md bg-surface p-3">
                                <summary className="cursor-pointer text-sm font-semibold">Advanced details</summary>
                                <div className="mt-3 space-y-3">
                                  <div className="flex flex-wrap gap-2">
                                    <Button
                                      className={primaryLocalAiButtonClass}
                                      disabled={
                                        isLocalAiBusy ||
                                        !localAiStatus?.localComfy?.runtimeInstalled ||
                                        !localAiGenerativeModelsReady ||
                                        !!localAiStatus?.localComfy?.running
                                      }
                                      onClick={handleStartLocalAiRuntime}
                                    >
                                      <PlayCircle
                                        size={16}
                                        className={localAiTask === 'runtime-start' ? 'animate-pulse' : ''}
                                      />
                                      {localAiTask === 'runtime-start' ? 'Starting...' : 'Start Runtime'}
                                    </Button>
                                    <Button
                                      className={secondaryLocalAiButtonClass}
                                      disabled={isLocalAiBusy || !localAiStatus?.localComfy?.running}
                                      onClick={handleStopLocalAiRuntime}
                                    >
                                      <RefreshCw
                                        size={16}
                                        className={localAiTask === 'runtime-stop' ? 'animate-spin' : ''}
                                      />
                                      {localAiTask === 'runtime-stop' ? 'Stopping...' : 'Stop Runtime'}
                                    </Button>
                                    <Button className={secondaryLocalAiButtonClass} disabled={isLocalAiBusy} onClick={resetLocalAiGenerationSettings}>
                                      <RefreshCw size={16} />
                                      Reset Generation Defaults
                                    </Button>
                                  </div>
                                  <div className="space-y-3 rounded-md bg-bg-primary p-3">
                                    <Text weight={TextWeights.semibold}>Generation settings</Text>
                                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                                      <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Steps
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="1"
                                        max="60"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ steps: Number(e.target.value) || 1 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        type="number"
                                        value={String(localAiGenerationSettings.steps)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        CFG
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="0"
                                        max="20"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ cfg: Number(e.target.value) || 0 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        step="0.1"
                                        type="number"
                                        value={String(localAiGenerationSettings.cfg)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Denoise
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="0"
                                        max="1"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ denoise: Number(e.target.value) || 0 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        step="0.05"
                                        type="number"
                                        value={String(localAiGenerationSettings.denoise)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Crop Target
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="512"
                                        max="2048"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ cropTarget: Number(e.target.value) || 512 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        step="64"
                                        type="number"
                                        value={String(localAiGenerationSettings.cropTarget)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Mask Blend
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="0"
                                        max="128"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ maskBlendPixels: Number(e.target.value) || 0 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        type="number"
                                        value={String(localAiGenerationSettings.maskBlendPixels)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        ControlNet
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        min="0"
                                        max="2"
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ controlnetStrength: Number(e.target.value) || 0 })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        step="0.05"
                                        type="number"
                                        value={String(localAiGenerationSettings.controlnetStrength)}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Sampler
                                      </Text>
                                      <Dropdown
                                        disabled={isLocalAiBusy}
                                        onChange={(value) => updateLocalAiGenerationSettings({ samplerName: value })}
                                        options={localAiSamplerOptions}
                                        value={localAiGenerationSettings.samplerName}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Scheduler
                                      </Text>
                                      <Dropdown
                                        disabled={isLocalAiBusy}
                                        onChange={(value) => updateLocalAiGenerationSettings({ scheduler: value })}
                                        options={localAiSchedulerOptions}
                                        value={localAiGenerationSettings.scheduler}
                                      />
                                      </label>
                                    </div>
                                    <div className="mt-3 grid grid-cols-1 xl:grid-cols-[1fr_220px] gap-3">
                                      <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Negative Prompt
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({ negativePrompt: e.target.value })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        type="text"
                                        value={localAiGenerationSettings.negativePrompt}
                                      />
                                    </label>
                                    <label className="space-y-1">
                                      <Text variant={TextVariants.small} weight={TextWeights.semibold}>
                                        Seed
                                      </Text>
                                      <Input
                                        disabled={isLocalAiBusy}
                                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                          updateLocalAiGenerationSettings({
                                            seed: e.target.value.trim() ? Number(e.target.value) : null,
                                          })
                                        }
                                        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.stopPropagation()}
                                        placeholder="Random"
                                        type="number"
                                        value={localAiGenerationSettings.seed === null ? '' : String(localAiGenerationSettings.seed)}
                                      />
                                      </label>
                                    </div>
                                  </div>
                                  <div className="space-y-3 rounded-md bg-bg-primary p-3">
                                    <Text weight={TextWeights.semibold}>Model files</Text>
                                    <div className="grid grid-cols-1 xl:grid-cols-3 gap-2">
                                      {localAiGenerativeModels.map((model) => (
                                        <Text
                                          key={model.id}
                                          as="div"
                                          color={model.installed && model.valid ? TextColors.success : TextColors.info}
                                          variant={TextVariants.small}
                                          className="min-w-0 rounded-md bg-surface px-3 py-2"
                                        >
                                          <span className="font-medium">{model.name}</span>
                                          <span className="block break-all text-text-secondary">
                                            {model.filename} · {model.installed ? formatBytes(model.sizeBytes) : 'Missing'}
                                          </span>
                                        </Text>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </details>

                              {localAiStatus?.localComfy?.lastError && (
                                <Text color={TextColors.error} className="block">
                                  Runtime: {localAiStatus.localComfy.lastError}
                                </Text>
                              )}
                              {isLocalAiGenerativeMessage && (
                                <Text
                                  color={lowerLocalAiMessage.includes('failed') ? TextColors.error : TextColors.accent}
                                  className="block"
                                >
                                  {localAiMessage}
                                </Text>
                              )}
                            </div>

                            <div className="p-4 bg-bg-primary rounded-lg border border-border-color space-y-3">
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                  <Text weight={TextWeights.semibold}>
                                    {localAiModel?.name || 'LaMa Inpainting'}
                                  </Text>
                                  <Text variant={TextVariants.small} className="block mt-1">
                                    {localAiModel?.filename || 'lama_fp16.onnx'} ·{' '}
                                    {localAiModel?.installed
                                      ? localAiModel.valid
                                        ? `Installed (${formatBytes(localAiModel.sizeBytes)})`
                                        : 'Installed but failed hash verification'
                                      : 'Not installed'}
                                  </Text>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  <Button
                                    className={secondaryLocalAiButtonClass}
                                    disabled={isLocalAiBusy}
                                    onClick={() => refreshLocalAiStatus(true, 'model-refresh')}
                                  >
                                    <RefreshCw
                                      size={16}
                                      className={localAiTask === 'model-refresh' ? 'animate-spin' : ''}
                                    />
                                    {localAiTask === 'model-refresh' ? 'Refreshing...' : 'Refresh'}
                                  </Button>
                                  <Button
                                    className={primaryLocalAiButtonClass}
                                    disabled={
                                      isLocalAiBusy ||
                                      !localAiStatus?.modelDirWritable ||
                                      (!!localAiModel?.installed && localAiModel.valid)
                                    }
                                    onClick={handleDownloadLocalAiModel}
                                  >
                                    <Download
                                      size={16}
                                      className={localAiTask === 'download' && !localAiDownloadProgress ? 'animate-pulse' : ''}
                                    />
                                    {localAiLamaDownloadButtonLabel}
                                  </Button>
                                  <Button
                                    className={secondaryLocalAiButtonClass}
                                    disabled={isLocalAiBusy || !localAiModel?.installed}
                                    onClick={handleDeleteLocalAiModel}
                                  >
                                    <Trash2 size={16} className={localAiTask === 'delete' ? 'animate-pulse' : ''} />
                                    {localAiTask === 'delete' ? 'Deleting...' : 'Delete'}
                                  </Button>
                                  <Button
                                    className={primaryLocalAiButtonClass}
                                    disabled={isLocalAiBusy || !localAiPrerequisitesReady}
                                    onClick={handleRunLocalAiSelfTest}
                                  >
                                    <PlayCircle
                                      size={16}
                                      className={localAiTask === 'self-test' ? 'animate-pulse' : ''}
                                    />
                                    {localAiTask === 'self-test' ? 'Testing...' : 'Run Test'}
                                  </Button>
                                </div>
                              </div>

                              <Text
                                as="div"
                                color={localAiReady ? TextColors.success : TextColors.info}
                                className="flex items-center gap-2"
                              >
                                {localAiReady ? <Wifi size={16} /> : <Info size={16} />}
                                {localAiStatusMessage}
                              </Text>

                              {localAiLamaTask && localAiTask !== 'download' && (
                                <div className="space-y-2" aria-live="polite">
                                  <div className="flex items-center gap-2">
                                    <RefreshCw size={14} className="animate-spin text-accent" />
                                    <Text variant={TextVariants.small} color={TextColors.accent}>
                                      {localAiTaskLabel}
                                    </Text>
                                  </div>
                                  <div className="h-2 overflow-hidden rounded-full bg-surface" role="progressbar">
                                    <div className="h-full w-1/3 animate-pulse rounded-full bg-accent" />
                                  </div>
                                </div>
                              )}

                              {localAiDownloadProgress && localAiTask === 'download' && (
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between gap-3">
                                    <Text variant={TextVariants.small} color={TextColors.accent}>
                                      {localAiDownloadLabel}
                                    </Text>
                                    {localAiDownloadPercent !== null && (
                                      <Text variant={TextVariants.small} color={TextColors.accent}>
                                        {localAiDownloadPercent}%
                                      </Text>
                                    )}
                                  </div>
                                  <div
                                    className="h-2 overflow-hidden rounded-full bg-surface"
                                    role="progressbar"
                                    aria-label="LaMa inpainting model download progress"
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-valuenow={localAiDownloadPercent ?? undefined}
                                  >
                                    <div
                                      className={clsx('h-full rounded-full bg-accent transition-all', {
                                        'animate-pulse': localAiDownloadPercent === null,
                                      })}
                                      style={{
                                        width: localAiDownloadPercent === null ? '35%' : `${localAiDownloadPercent}%`,
                                      }}
                                    />
                                  </div>
                                </div>
                              )}

                              {localAiStatus && !localAiStatus.isWindows && (
                                <Text color={TextColors.error} className="block">
                                  Local GPU is currently enabled for Windows x64 only.
                                </Text>
                              )}
                              {localAiStatus && !localAiStatus.cudaAvailable && (
                                <Text color={TextColors.error} className="block">
                                  No NVIDIA CUDA GPU was detected. Install or update the NVIDIA driver from{' '}
                                  <a
                                    className="font-semibold text-accent hover:underline"
                                    href={NVIDIA_DRIVER_DOWNLOAD_URL}
                                    rel="noopener noreferrer"
                                    target="_blank"
                                  >
                                    NVIDIA Driver Downloads
                                  </a>
                                  , then restart RapidRAW.
                                </Text>
                              )}
                              {localAiStatus?.cudaProviderError && !missingLocalAiRuntimeDependencies.length && (
                                <Text color={TextColors.error} className="block">
                                  CUDA provider: {localAiStatus.cudaProviderError}
                                </Text>
                              )}
                              {localAiStatus?.modelDirError && (
                                <Text color={TextColors.error} className="block">
                                  Model folder: {localAiStatus.modelDirError}
                                </Text>
                              )}
                              {isLocalAiLamaMessage && (
                                <Text
                                  color={lowerLocalAiMessage.includes('failed') ? TextColors.error : TextColors.accent}
                                  className="block"
                                >
                                  {localAiMessage}
                                </Text>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      )}

                      {aiProvider === 'ai-connector' && (
                        <motion.div
                          key="ai-connector"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                        >
                          <div className="space-y-8">
                            <div>
                              <Text variant={TextVariants.heading}>{t('settings.processing.ai.connector.title')}</Text>
                              <Text className="mt-1">{t('settings.processing.ai.connector.description')}</Text>
                              <Text as="ul" className="mt-3 space-y-1 list-disc list-inside">
                                <li>{t('settings.processing.ai.connector.feature1')}</li>
                                <li>{t('settings.processing.ai.connector.feature2')}</li>
                                <li>{t('settings.processing.ai.connector.feature3')}</li>
                              </Text>
                            </div>
                            <SettingItem
                              label={t('settings.processing.ai.connector.address')}
                              description={t('settings.processing.ai.connector.addressDesc')}
                            >
                              <div className="flex items-center gap-2">
                                <Input
                                  className="grow"
                                  id="ai-connector-address"
                                  onBlur={() =>
                                    onSettingsChange({ ...appSettings, aiConnectorAddress: aiConnectorAddress })
                                  }
                                  onChange={(e: any) => setAiConnectorAddress(e.target.value)}
                                  onKeyDown={(e: any) => e.stopPropagation()}
                                  placeholder="127.0.0.1:8188"
                                  type="text"
                                  value={aiConnectorAddress}
                                  bgClassName="bg-bg-primary"
                                />
                                <Button
                                  className="w-32"
                                  disabled={testStatus.testing || !aiConnectorAddress}
                                  onClick={handleTestConnection}
                                >
                                  {testStatus.testing
                                    ? t('settings.processing.ai.connector.testing')
                                    : t('settings.processing.ai.connector.test')}
                                </Button>
                              </div>
                              {testStatus.message && (
                                <Text
                                  color={testStatus.success ? TextColors.success : TextColors.error}
                                  className="mt-2 flex items-center gap-2"
                                >
                                  {testStatus.success === true && <Wifi size={16} />}
                                  {testStatus.success === false && <WifiOff size={16} />}
                                  {testStatus.message}
                                </Text>
                              )}
                            </SettingItem>
                          </div>
                        </motion.div>
                      )}

                      {aiProvider === 'cloud' && (
                        <motion.div
                          key="cloud"
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                        >
                          <Text variant={TextVariants.heading}>{t('settings.processing.ai.cloud.title')}</Text>
                          <Text className="mt-1">{t('settings.processing.ai.cloud.description')}</Text>
                          <Text as="ul" className="mt-3 space-y-1 list-disc list-inside">
                            <li>{t('settings.processing.ai.cloud.feature1')}</li>
                            <li>{t('settings.processing.ai.cloud.feature2')}</li>
                            <li>{t('settings.processing.ai.cloud.feature3')}</li>
                          </Text>

                          <div className="mt-8">
                            <Show when="signed-in">
                              <div className="p-6 bg-bg-primary rounded-xl border border-border-color shadow-inner">
                                <CloudDashboard />
                              </div>
                            </Show>
                            <Show when="signed-out">
                              <div className="w-full max-w-md">
                                <SignIn
                                  routing="hash"
                                  fallbackRedirectUrl="/"
                                  forceRedirectUrl="/"
                                  appearance={{
                                    variables: {
                                      colorBackground: 'transparent',
                                      colorInput: 'transparent',
                                      colorForeground: 'inherit',
                                      colorInputForeground: 'inherit',
                                      colorTextOnPrimaryBackground: 'inherit',
                                      colorPrimaryForeground: 'inherit',
                                      colorBorder: 'transparent',
                                      colorShadow: 'none',
                                      colorNeutral: 'inherit',
                                    },
                                    elements: {
                                      rootBox: '',

                                      cardBox: '!shadow-none !m-0 !p-0 !rounded-none',

                                      card: '!bg-transparent !border-none !shadow-none !py-0 !px-1 !rounded-none',

                                      header: '!hidden',

                                      formFieldLabel: '!text-base !font-semibold !text-text-primary !block !mb-2',

                                      formFieldAction:
                                        '!text-text-secondary hover:!text-text-primary !transition-colors !no-underline hover:!underline',

                                      formFieldInput:
                                        '!bg-bg-primary !border !border-border-color !text-text-primary focus:!border-accent focus:!ring-1 focus:!ring-accent !rounded-md !px-3 !py-2',

                                      formButtonPrimary:
                                        '!bg-accent !text-button-text hover:!bg-accent/90 !shadow-none !transition-colors !rounded-md !mt-4 !py-2',

                                      footer:
                                        '!bg-transparent !p-0 !mt-4 opacity-50 hover:opacity-100 transition-opacity',
                                      footerAction: '!hidden',

                                      identityPreview: '!bg-bg-primary !border !border-border-color !rounded-md !mb-4',
                                      identityPreviewText: '!text-text-primary !font-medium',
                                      identityPreviewEditButtonIcon:
                                        '!text-text-secondary hover:!text-text-primary !transition-colors',
                                    },
                                  }}
                                />
                                <div className="mt-6">
                                  <Text variant={TextVariants.small}>
                                    {t('settings.processing.ai.cloud.signedOut.noAccount')}{' '}
                                    <button
                                      onClick={() =>
                                        void openExternalUrl('https://www.getrapidraw.com/dashboard', {
                                          allowedHosts: ['www.getrapidraw.com'],
                                        })
                                      }
                                      className="text-accent hover:underline focus:outline-none"
                                    >
                                      {t('settings.processing.ai.cloud.signedOut.signup')}
                                    </button>
                                  </Text>
                                </div>
                              </div>
                            </Show>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.data.title')}
                  </Text>
                  <div className="space-y-8">
                    <DataActionItem
                      buttonAction={handleClearSidecars}
                      buttonText={t('settings.data.clearSidecarsButton')}
                      description={
                        <Text as="span" variant={TextVariants.small}>
                          {t('settings.data.clearSidecarsDesc')}{' '}
                          <code className="bg-bg-primary px-1 rounded-sm text-text-primary">.rrdata</code> files
                          (containing your edits) within your root folders:
                          <span className="block font-mono bg-bg-primary p-2 rounded-sm mt-2 break-all border border-border-color whitespace-pre-wrap">
                            {effectiveRootPaths.length > 0
                              ? effectiveRootPaths.join('\n')
                              : t('settings.data.noFolders')}
                          </span>
                        </Text>
                      }
                      disabled={effectiveRootPaths.length === 0}
                      icon={<Trash2 size={16} className="mr-2" />}
                      isProcessing={isClearing}
                      message={clearMessage}
                      title={t('settings.data.clearSidecars')}
                    />

                    <DataActionItem
                      buttonAction={handleClearCache}
                      buttonText={t('settings.data.clearThumbnailButton')}
                      description={t('settings.data.clearThumbnailDesc')}
                      icon={<Trash2 size={16} className="mr-2" />}
                      isProcessing={isClearingCache}
                      message={cacheClearMessage}
                      title={t('settings.data.clearThumbnail')}
                    />

                    <DataActionItem
                      buttonAction={async () => {
                        if (logPath && !logPathLoading && !logPathError) {
                          await invoke(Invokes.ShowInFinder, { path: logPath });
                        }
                      }}
                      buttonText={t('settings.data.logsButton')}
                      description={
                        <Text as="span" variant={TextVariants.small}>
                          {t('settings.data.logsDesc')}
                          <span className="block font-mono bg-bg-primary p-2 rounded-sm mt-2 break-all border border-border-color">
                            {logPathLoading
                              ? t('settings.data.loading')
                              : logPathError
                                ? t('settings.data.statuses.failedToGetPath')
                                : logPath}
                          </span>
                        </Text>
                      }
                      disabled={logPathLoading || logPathError || !logPath}
                      icon={<ExternalLinkIcon size={16} className="mr-2" />}
                      isProcessing={false}
                      message=""
                      title={t('settings.data.logs')}
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {activeCategory === 'googlePhotos' && (
              <motion.div
                key="googlePhotos"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-10"
              >
                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    Google Photos
                  </Text>
                  <div className="space-y-8">
                    <SettingItem
                      label="Integration"
                      description="Uses Google OAuth for desktop apps with a system browser, PKCE, a local loopback redirect, and app-created Google Photos data scopes."
                    >
                      <Switch
                        checked={googlePhotosIntegrationEnabled}
                        id="google-photos-integration-toggle"
                        label="Enable Google Photos"
                        onChange={handleGooglePhotosIntegrationToggle}
                      />
                      {!googlePhotosIntegrationEnabled && hasSavedGooglePhotosAlbum && (
                        <Text variant={TextVariants.small} className="block mt-3">
                          Sync is off. The Google Photos album "{savedGooglePhotosAlbumTitle}" is hidden in RapidRAW, but
                          the saved album settings are kept for when you turn sync back on.
                        </Text>
                      )}
                    </SettingItem>

                    <div className="grid grid-cols-1 min-[900px]:grid-cols-2 gap-4">
                      <SettingItem
                        label="OAuth Client ID"
                        description="Use a Google Cloud OAuth Desktop client ID from the project with Photos Library API enabled."
                      >
                        <Input
                          bgClassName="bg-bg-primary"
                          id="google-photos-client-id"
                          onBlur={() => saveGooglePhotosCredentials()}
                          onChange={(e: any) => setGooglePhotosClientId(e.target.value)}
                          onKeyDown={(e: any) => e.stopPropagation()}
                          placeholder="1234567890-abc.apps.googleusercontent.com"
                          type="text"
                          value={googlePhotosClientId}
                        />
                      </SettingItem>

                      <SettingItem
                        label="OAuth Client Secret"
                        description="Optional for public desktop clients. Enter it only if your Google OAuth client requires it."
                      >
                        <Input
                          bgClassName="bg-bg-primary"
                          id="google-photos-client-secret"
                          onBlur={() => saveGooglePhotosCredentials()}
                          onChange={(e: any) => setGooglePhotosClientSecret(e.target.value)}
                          onKeyDown={(e: any) => e.stopPropagation()}
                          placeholder="Optional"
                          type="password"
                          value={googlePhotosClientSecret}
                        />
                      </SettingItem>
                    </div>

                    <div className="flex flex-wrap gap-3">
                      <Button
                        onClick={handleGooglePhotosLogin}
                        disabled={googlePhotosBusy || !googlePhotosClientId.trim()}
                      >
                        <Cloud size={16} className={googlePhotosBusy ? 'animate-pulse' : ''} />
                        {googlePhotosStatus?.authenticated ? 'Reconnect Google' : 'Sign in with Google'}
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={handleGooglePhotosDisconnect}
                        disabled={googlePhotosBusy || !googlePhotosStatus?.authenticated}
                      >
                        <WifiOff size={16} />
                        Disconnect
                      </Button>
                    </div>

                    <div className="p-4 bg-bg-primary rounded-lg border border-border-color space-y-3">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <Text variant={TextVariants.heading}>Connection Status</Text>
                          <Text variant={TextVariants.small}>
                            {googlePhotosIntegrationEnabled ? 'Sync on' : 'Sync off'}
                            {' · '}
                            {googlePhotosStatus?.authenticated ? 'Connected' : 'Not connected'}
                            {googlePhotosStatus?.syncedCount ? ` · ${googlePhotosStatus.syncedCount} synced` : ''}
                          </Text>
                        </div>
                        {googlePhotosStatus?.authenticated ? (
                          <Wifi size={18} className="text-green-400" />
                        ) : (
                          <WifiOff size={18} className="text-text-secondary" />
                        )}
                      </div>
                      {googlePhotosMessage && (
                        <Text color={isGooglePhotosErrorMessage(googlePhotosMessage) ? TextColors.error : TextColors.accent}>
                          {googlePhotosMessage}
                        </Text>
                      )}
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    Album Management
                  </Text>
                  <div className="space-y-8">
                    <SettingItem
                      label="RapidRaw Album"
                      description="RapidRAW can create and manage one app-created Google Photos album. Google Photos API access is limited to albums and media created by this app."
                    >
                      <div className="flex flex-col min-[760px]:flex-row gap-3">
                        <Input
                          className="grow"
                          bgClassName="bg-bg-primary"
                          id="google-photos-album-title"
                          onBlur={() =>
                            onSettingsChange({ ...appSettings, googlePhotosAlbumTitle: googlePhotosAlbumTitleInput })
                          }
                          onChange={(e: any) => setGooglePhotosAlbumTitleInput(e.target.value)}
                          onKeyDown={(e: any) => e.stopPropagation()}
                          placeholder="RapidRaw"
                          type="text"
                          value={googlePhotosAlbumTitleInput}
                        />
                        <Button
                          className="min-[760px]:w-38"
                          disabled={googlePhotosBusy || !googlePhotosStatus?.authenticated}
                          onClick={handleGooglePhotosCreateAlbum}
                        >
                          <Plus size={16} />
                          Create
                        </Button>
                        <Button
                          className="min-[760px]:w-38"
                          variant="ghost"
                          disabled={
                            googlePhotosBusy || !googlePhotosStatus?.authenticated || !appSettings?.googlePhotosAlbumId
                          }
                          onClick={handleGooglePhotosRenameAlbum}
                        >
                          <FileEdit size={16} />
                          Rename
                        </Button>
                      </div>
                    </SettingItem>

                    <Text variant={TextVariants.small} className="block">
                      Active album:{' '}
                      <span className="font-mono text-text-primary">
                        {appSettings?.googlePhotosAlbumTitle || googlePhotosStatus?.albumTitle || 'RapidRaw'}
                      </span>
                    </Text>
                  </div>
                </div>
              </motion.div>
            )}

            {activeCategory === 'shortcuts' && (
              <motion.div
                key="shortcuts"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-10"
              >
                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.controls.title')}
                  </Text>
                  <div className="space-y-8">
                    <div>
                      <Text variant={TextVariants.heading} className="mb-2">
                        {t('settings.controls.optimization')}
                      </Text>
                      <Text variant={TextVariants.small} className="mb-4">
                        {t('settings.controls.optimizationDesc')}
                      </Text>
                      <CanvasInputModeSwitch
                        mode={(appSettings?.canvasInputMode as 'mouse' | 'trackpad') || 'mouse'}
                        onModeChange={(value) => onSettingsChange({ ...appSettings, canvasInputMode: value })}
                      />
                    </div>

                    <SettingItem label={t('settings.controls.zoom')} description={t('settings.controls.zoomDesc')}>
                      <Slider
                        label={t('settings.controls.speed')}
                        min={0.1}
                        max={3.0}
                        step={0.1}
                        value={appSettings?.zoomSpeedMultiplier ?? 1.0}
                        defaultValue={1.0}
                        onChange={(e: any) =>
                          onSettingsChange({ ...appSettings, zoomSpeedMultiplier: parseFloat(e.target.value) })
                        }
                        fillOrigin="min"
                      />
                    </SettingItem>
                  </div>
                </div>

                <div className="p-6 bg-surface rounded-xl shadow-md">
                  <Text variant={TextVariants.title} color={TextColors.accent} className="mb-8">
                    {t('settings.controls.keyboardTitle')}
                  </Text>
                  <div className="space-y-8">
                    {' '}
                    {KEYBIND_SECTIONS.map((section) => {
                      const sectionDefs = KEYBIND_DEFINITIONS.filter((d) => d.section === section.id);
                      const userKb = appSettings?.keybinds || {};
                      return (
                        <div key={section.id}>
                          <Text variant={TextVariants.heading}>{t(section.label as any)}</Text>
                          <div className="divide-y divide-border-color">
                            {sectionDefs.map((def) => (
                              <KeybindRow
                                key={def.action}
                                def={def}
                                currentCombo={userKb[def.action]}
                                osPlatform={osPlatform}
                                onSave={handleKeybindSave}
                                recordingAction={recordingAction}
                                onStartRecording={setRecordingAction}
                                isConflicting={conflictingKeys.has(def.action)}
                              />
                            ))}
                          </div>
                        </div>
                      );
                    })}
                    <div className="flex justify-end mt-6">
                      <Button variant="ghost" onClick={() => onSettingsChange({ ...appSettings, keybinds: {} })}>
                        {t('settings.controls.resetDefaults')}
                      </Button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </>
  );
}
