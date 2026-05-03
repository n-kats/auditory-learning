import { ApiError } from "../api";

export function buildErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "予期しないエラーが発生しました。";
}

export function buildPaperLabel(sourceUrl: string): string | null {
  try {
    const parsedUrl = new URL(sourceUrl);
    const arxivMatch = parsedUrl.pathname.match(/\/pdf\/([^/]+)(?:\.pdf)?$/);
    if (arxivMatch?.[1]) {
      return `arXiv ${arxivMatch[1]}`;
    }
    const openReviewId = parsedUrl.searchParams.get("id");
    if (openReviewId) {
      return `OpenReview ${openReviewId}`;
    }
    const lastPathPart = parsedUrl.pathname.split("/").filter(Boolean).pop();
    if (lastPathPart) {
      return lastPathPart;
    }
  } catch {
    return null;
  }
  return null;
}
