import { useEffect, useRef, useState } from "react";

import { fetchDefaultPrompt, isNetworkFetchError } from "../api";

type UsePromptTemplateResult = {
  defaultExplainPromptText: string;
  defaultSpeekPromptText: string;
  draftExplainPromptText: string;
  draftSpeekPromptText: string;
  setDraftExplainPromptText: (value: string) => void;
  setDraftSpeekPromptText: (value: string) => void;
};

export function usePromptTemplate(): UsePromptTemplateResult {
  const [defaultExplainPromptText, setDefaultExplainPromptText] = useState("");
  const [defaultSpeekPromptText, setDefaultSpeekPromptText] = useState("");
  const [draftExplainPromptText, setDraftExplainPromptText] = useState("");
  const [draftSpeekPromptText, setDraftSpeekPromptText] = useState("");
  const explainPromptDirtyRef = useRef(false);
  const speekPromptDirtyRef = useRef(false);
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
        setDefaultExplainPromptText(response.prompt_explain_text);
        setDefaultSpeekPromptText(response.prompt_speek_text);
        if (!explainPromptDirtyRef.current) {
          setDraftExplainPromptText(response.prompt_explain_text);
        }
        if (!speekPromptDirtyRef.current) {
          setDraftSpeekPromptText(response.prompt_speek_text);
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
        if (!explainPromptDirtyRef.current) {
          setDraftExplainPromptText("");
        }
        if (!speekPromptDirtyRef.current) {
          setDraftSpeekPromptText("");
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
    defaultExplainPromptText,
    defaultSpeekPromptText,
    draftExplainPromptText,
    draftSpeekPromptText,
    setDraftExplainPromptText: (value: string) => {
      explainPromptDirtyRef.current = true;
      setDraftExplainPromptText(value);
    },
    setDraftSpeekPromptText: (value: string) => {
      speekPromptDirtyRef.current = true;
      setDraftSpeekPromptText(value);
    },
  };
}
