import { useEffect, useState } from "react";

import { fetchFavorites, toggleFavorite, type FavoritePaperItem } from "../api";
import { buildPaperLabel } from "../utils/appText";

export function FavoritesPage() {
  const [items, setItems] = useState<FavoritePaperItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pendingFavoriteKey, setPendingFavoriteKey] = useState<string | null>(null);

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

  const handleToggleFavorite = async (requestId: string, pageNum: number) => {
    const favoriteKey = `${requestId}:${pageNum}`;
    setPendingFavoriteKey(favoriteKey);
    try {
      const response = await toggleFavorite(requestId, pageNum);
      setItems((current) => {
        if (!response.favorited) {
          return current.filter((item) => !(item.request_id === requestId && item.favorite_page_num === response.page_num));
        }
        return current;
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "favorites の更新に失敗しました。");
    } finally {
      setPendingFavoriteKey(null);
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
              <li key={`${item.request_id}:${item.favorite_page_num}`} className="session-row">
                <button
                  className="session-open-button is-current"
                  type="button"
                  onClick={() => void handleToggleFavorite(item.request_id, item.favorite_page_num)}
                  disabled={pendingFavoriteKey === `${item.request_id}:${item.favorite_page_num}`}
                >
                  解除
                </button>
                <div className="session-item is-current">
                  <div className="session-item-main">
                    <p className="directory-item-url">{buildPaperLabel(item.source_url) ?? item.source_url}</p>
                    <p className="directory-item-meta">
                      session: {item.request_id} / favorite: p. {item.favorite_page_num} / document: {item.page_num}
                    </p>
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
