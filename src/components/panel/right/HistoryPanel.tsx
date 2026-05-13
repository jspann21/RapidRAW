import { Clock3 } from 'lucide-react';
import clsx from 'clsx';
import { useShallow } from 'zustand/react/shallow';
import { useEditorStore } from '../../../store/useEditorStore';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';

const formatTime = (timestamp: string) => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};

export default function HistoryPanel() {
  const { history, historyIndex, goToHistoryIndex, selectedImage } = useEditorStore(
    useShallow((state) => ({
      history: state.history,
      historyIndex: state.historyIndex,
      goToHistoryIndex: state.goToHistoryIndex,
      selectedImage: state.selectedImage,
    })),
  );

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 flex justify-between items-center shrink-0 border-b border-surface">
        <Text variant={TextVariants.title}>History</Text>
        <Clock3 size={18} className="text-text-secondary" />
      </div>

      <div className="grow min-h-0 overflow-y-auto p-3">
        {!selectedImage || history.length === 0 ? (
          <div className="h-full flex items-center justify-center px-4 text-center">
            <Text color={TextColors.secondary}>No edit history for this photo.</Text>
          </div>
        ) : (
          <div className="space-y-1">
            {history.map((entry, index) => {
              const isCurrent = index === historyIndex;
              const isFuture = index > historyIndex;
              return (
                <button
                  key={entry.id}
                  className={clsx(
                    'w-full text-left px-3 py-2 rounded-md transition-colors outline-none focus:ring-1 focus:ring-accent',
                    isCurrent
                      ? 'bg-surface text-text-primary'
                      : 'text-text-secondary hover:bg-card-active hover:text-text-primary',
                    isFuture && 'opacity-70',
                  )}
                  onClick={() => goToHistoryIndex(index)}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <div
                      className={clsx(
                        'w-1.5 h-1.5 rounded-full shrink-0',
                        isCurrent ? 'bg-primary' : isFuture ? 'bg-text-secondary/40' : 'bg-text-secondary',
                      )}
                    />
                    <Text
                      as="span"
                      color={isCurrent ? TextColors.primary : TextColors.secondary}
                      weight={isCurrent ? TextWeights.bold : TextWeights.medium}
                      className="truncate"
                    >
                      {entry.label}
                    </Text>
                    <Text as="span" variant={TextVariants.small} color={TextColors.secondary} className="ml-auto">
                      {formatTime(entry.timestamp)}
                    </Text>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
