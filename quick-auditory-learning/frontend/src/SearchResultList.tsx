import type { ReactNode } from "react";

type SearchResultItemProps = {
  id: string;
  title: string;
  paperIdLabel: string;
  meta: ReactNode;
  sourceModes: string[];
  isSelected: boolean;
  isQueued: boolean;
  isReplayed: boolean;
  isFavorite: boolean;
  canInteract: boolean;
  onToggleQueue: () => void;
  onToggleFavorite: () => void;
};

export function SearchResultListItem({
  id,
  title,
  paperIdLabel,
  meta,
  sourceModes,
  isSelected,
  isQueued,
  isReplayed,
  isFavorite,
  canInteract,
  onToggleQueue,
  onToggleFavorite,
}: SearchResultItemProps) {
  return (
    <li key={id} className={`result-item${isSelected ? " is-selected" : ""}${isQueued ? " is-queued" : ""}${isReplayed ? " is-replayed" : ""}`}>
      <div className="result-row">
        <button type="button" className={`result-next-button${isQueued ? " is-queued" : ""}`} disabled={!canInteract} onClick={onToggleQueue}>
          次に再生
        </button>
        <div className={`result-card${isSelected ? " is-selected" : ""}${isQueued ? " is-queued" : ""}${isReplayed ? " is-replayed" : ""}`}>
          <div className="result-card-head">
            <div className="result-main">
              <div className="result-title-line">
                <p className="paper-id">{paperIdLabel}</p>
                {isReplayed ? <span className="status-badge is-replayed">再生済み</span> : null}
                <h3>{title}</h3>
              </div>
              <p className="meta result-meta">{meta}</p>
              {sourceModes.length > 0 ? (
                <div className="result-source-modes" aria-label="検索方法">
                  {sourceModes.map((mode) => (
                    <span key={mode} className="source-mode-chip">
                      {mode}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
            <button
              type="button"
              className={`current-session-action-button result-favorite-button${isFavorite ? " is-active" : ""}`}
              onClick={onToggleFavorite}
              aria-label={isFavorite ? "お気に入り解除" : "お気に入り"}
              title={isFavorite ? "お気に入り解除" : "お気に入り"}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 20.4 4.6 13c-1.9-1.9-1.9-4.9 0-6.8 1.9-1.9 4.9-1.9 6.8 0l.6.6.6-.6c1.9-1.9 4.9-1.9 6.8 0 1.9 1.9 1.9 4.9 0 6.8Z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                  fill="none"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </li>
  );
}

type SearchResultListProps = {
  items: SearchResultItemProps[];
};

export function SearchResultList({ items }: SearchResultListProps) {
  return <ul className="result-list">{items.map((item) => <SearchResultListItem key={item.id} {...item} />)}</ul>;
}
