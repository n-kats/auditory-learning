export type ControlIconKind = "stop" | "play" | "next" | "regenerate" | "upload";

export function ControlIcon({ kind }: { kind: ControlIconKind }) {
  switch (kind) {
    case "stop":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="6.5" y="6.5" width="11" height="11" rx="1.4" fill="currentColor" />
        </svg>
      );
    case "play":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M8 6.8v10.4c0 .8.9 1.3 1.6.9l8.6-5.2c.7-.4.7-1.4 0-1.8L9.6 5.9c-.7-.4-1.6.1-1.6.9Z" fill="currentColor" />
        </svg>
      );
    case "next":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M8 6.8v10.4c0 .8.9 1.3 1.6.9l6.7-4c.7-.4.7-1.4 0-1.8l-6.7-4c-.7-.4-1.6.1-1.6.9Z"
            fill="currentColor"
          />
          <path d="M17.5 6.5v11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
        </svg>
      );
    case "regenerate":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6.9 8.2A7 7 0 0 1 17.3 6.1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
          <path d="M17.3 6.1V9.6h-3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <path d="M17.2 15.8A7 7 0 0 1 6.8 17.9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
          <path d="M6.8 17.9v-3.5h3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      );
    case "upload":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 4.8v9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
          <path d="M8.6 8.5 12 5.1l3.4 3.4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <path d="M6.5 14.6v1.5c0 .9.7 1.6 1.6 1.6h7.8c.9 0 1.6-.7 1.6-1.6v-1.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
        </svg>
      );
  }
}
