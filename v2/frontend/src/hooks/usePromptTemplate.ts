import { useEffect, useRef, useState } from "react";

import { fetchDefaultPrompt, isNetworkFetchError } from "../api";

type UsePromptTemplateResult = {
  defaultExplainPromptText: string;
  defaultSpeakPromptText: string;
  defaultModelName: string;
  defaultReasoningEffort: string;
  draftExplainPromptText: string;
  draftSpeakPromptText: string;
  draftModelName: string;
  draftReasoningEffort: string;
  setDraftExplainPromptText: (value: string) => void;
  setDraftSpeakPromptText: (value: string) => void;
  setDraftModelName: (value: string) => void;
  setDraftReasoningEffort: (value: string) => void;
};

export function usePromptTemplate(): UsePromptTemplateResult {
  const [defaultExplainPromptText, setDefaultExplainPromptText] = useState("");
  const [defaultSpeakPromptText, setDefaultSpeakPromptText] = useState("");
  const [defaultModelName, setDefaultModelName] = useState("");
  const [defaultReasoningEffort, setDefaultReasoningEffort] = useState("");
  const [draftExplainPromptText, setDraftExplainPromptText] = useState("");
  const [draftSpeakPromptText, setDraftSpeakPromptText] = useState("");
  const [draftModelName, setDraftModelName] = useState("");
  const [draftReasoningEffort, setDraftReasoningEffort] = useState("");
  const explainPromptDirtyRef = useRef(false);
  const speakPromptDirtyRef = useRef(false);
  const modelNameDirtyRef = useRef(false);
  const reasoningEffortDirtyRef = useRef(false);
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
        setDefaultSpeakPromptText(response.prompt_speak_text);
        setDefaultModelName(response.model_name);
        setDefaultReasoningEffort(response.reasoning_effort);
        if (!explainPromptDirtyRef.current) {
          setDraftExplainPromptText(response.prompt_explain_text);
        }
        if (!speakPromptDirtyRef.current) {
          setDraftSpeakPromptText(response.prompt_speak_text);
        }
        if (!modelNameDirtyRef.current) {
          setDraftModelName(response.model_name);
        }
        if (!reasoningEffortDirtyRef.current) {
          setDraftReasoningEffort(response.reasoning_effort);
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
        if (!speakPromptDirtyRef.current) {
          setDraftSpeakPromptText("");
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
    defaultSpeakPromptText,
    defaultModelName,
    defaultReasoningEffort,
    draftExplainPromptText,
    draftSpeakPromptText,
    draftModelName,
    draftReasoningEffort,
    setDraftExplainPromptText: (value: string) => {
      explainPromptDirtyRef.current = true;
      setDraftExplainPromptText(value);
    },
    setDraftSpeakPromptText: (value: string) => {
      speakPromptDirtyRef.current = true;
      setDraftSpeakPromptText(value);
    },
    setDraftModelName: (value: string) => {
      modelNameDirtyRef.current = true;
      setDraftModelName(value);
    },
    setDraftReasoningEffort: (value: string) => {
      reasoningEffortDirtyRef.current = true;
      setDraftReasoningEffort(value);
    },
  };
}
