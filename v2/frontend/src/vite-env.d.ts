/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AUDITORY_LEARNING_V2_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
