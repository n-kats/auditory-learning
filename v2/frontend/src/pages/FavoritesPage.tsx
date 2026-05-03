import { useEffect, useState } from "react";

import { fetchFavorites, toggleFavorite, type FavoritePaperItem } from "../api";

export function FavoritesPage() {
  const [items, setItems] = useState<FavoritePaperItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pendingRequestId, setPendingRequestId] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;

    const load = async () => {
      setIsLoading(true);
      try {
        const response = await fetchFavorites(50);
        if (!canceled) {
          setItems(response.items);
          setError(null);
        }
      } catch (caught) {
        if (!canceled) {
          setError(caught instanceof Error ? caught.message : "favorites の読み込みに失敗しました。");
          setItems([]);
        }
      } finally {
        if (!canceled) {
          setIsLoading(false);
        }
      }
    };

    void load();

    return () => {
      canceled = true;
    };
  }, []);

  const handleToggleFavorite = async (requestId: string) => {
    setPendingRequestId(requestId);
    try {
      const response = await toggleFavorite(requestId);
      setItems((current) => {
        if (!response.favorited) {
          return current.filter((item) => item.request_id !== requestId);
        }
        return current;
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "favorites の更新に失敗しました。");
    } finally {
      setPendingRequestId(null);
    }
  };

  return (
    <main className="directory-shell">
      <section className="card directory-hero">
        <div className="directory-hero-head">
          <div>
            <p className="section-eyebrow">favorites</p>
            <h1 className="directory-title">お気に入り確認・管理</h1>
          </div>
        </div>

        {error ? <div className="directory-error">{error}</div> : null}

        {isLoading ? (
          <p className="directory-empty">読み込み中...</p>
        ) : items.length > 0 ? (
          <ul className="session-list">
            {items.map((item) => (
              <li key={item.request_id} className="session-row">
                <button
                  className="session-open-button is-current"
                  type="button"
                  onClick={() => void handleToggleFavorite(item.request_id)}
                  disabled={pendingRequestId === item.request_id}
                >
                  解除
                </button>
                <div className="session-item is-current">
                  <div className="session-item-main">
                    <p className="directory-item-url">{item.source_url}</p>
                    <p className="directory-item-meta">p. {item.current_page ?? 1} / {item.page_num ?? 1}</p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="directory-empty">まだありません。</p>
        )}
      </section>
    </main>
  );
}
