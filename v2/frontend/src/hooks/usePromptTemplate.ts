import { useEffect, useRef, useState } from "react";

import { fetchDefaultPrompt, isNetworkFetchError } from "../api";

type UsePromptTemplateResult = {
  defaultPromptText: string;
  draftPromptText: string;
  setDraftPromptText: (value: string) => void;
};

export function usePromptTemplate(): UsePromptTemplateResult {
  const [defaultPromptText, setDefaultPromptText] = useState("");
  const [draftPromptText, setDraftPromptText] = useState("");
  const promptTemplateDirtyRef = useRef(false);
  const promptLoadRetryRef = useRef<number | null>(null);

  useEffect(() => {
    let canceled = false;

    const loadPromptTemplate = async () => {
      try {
        const response = await fetchDefaultPrompt();
        if (canceled) {
          return;
        }
        if (promptLoadRetryRef.current !== null) {
          window.clearTimeout(promptLoadRetryRef.current);
          promptLoadRetryRef.current = null;
        }
        setDefaultPromptText(response.prompt_text);
        if (!promptTemplateDirtyRef.current) {
          setDraftPromptText(response.prompt_text);
        }
      } catch (error) {
        if (canceled) {
          return;
        }
        if (isNetworkFetchError(error)) {
          promptLoadRetryRef.current = window.setTimeout(() => {
            void loadPromptTemplate();
          }, 1000);
          return;
        }
        if (!promptTemplateDirtyRef.current) {
          setDraftPromptText("");
        }
      }
    };

    void loadPromptTemplate();

    return () => {
      canceled = true;
      if (promptLoadRetryRef.current !== null) {
        window.clearTimeout(promptLoadRetryRef.current);
        promptLoadRetryRef.current = null;
      }
    };
  }, []);

  return {
    defaultPromptText,
    draftPromptText,
    setDraftPromptText: (value: string) => {
      promptTemplateDirtyRef.current = true;
      setDraftPromptText(value);
    },
  };
}
