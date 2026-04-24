import { createOpenAI, type OpenAIProvider } from "@ai-sdk/openai";
import { createXai } from "@ai-sdk/xai";
import { generateText } from "ai";
import type { ModelInfo, ReasoningEffort } from "../types/index";
import { getReasoningEffortForModel } from "../utils/settings";
import { getEffectiveReasoningEffort, getModelInfo, normalizeModelId } from "./models";

export type XaiProvider = ReturnType<typeof createXai>;
export type XaiChatModel = ReturnType<XaiProvider>;
export type XaiResponsesModel = ReturnType<XaiProvider["responses"]>;
export type GrokRuntimeModel = XaiChatModel | XaiResponsesModel;

// When using a custom base URL (OpenRouter, etc.), the provider is an
// OpenAI-compatible instance that we cast to XaiProvider for type compat.
// This ref keeps the original typed OpenAI provider for .chat() access.
let _openaiProvider: OpenAIProvider | null = null;

const DEFAULT_BASE_URL = "https://api.x.ai/v1";
const DEFAULT_TITLE_MODEL = "grok-4-1-fast-non-reasoning";

export interface GeneratedTitle {
  title: string;
  modelId: string;
  usage?: {
    totalTokens?: number;
    inputTokens?: number;
    outputTokens?: number;
  };
}

export interface ResolvedModelRuntime {
  model: GrokRuntimeModel;
  modelId: string;
  modelInfo?: ModelInfo;
  providerOptions?: {
    xai: {
      reasoningEffort: ReasoningEffort;
    };
  };
}

export function isCustomBaseURL(baseURL?: string): boolean {
  const effective = baseURL || process.env.GROK_BASE_URL || DEFAULT_BASE_URL;
  return effective !== DEFAULT_BASE_URL;
}

export function createProvider(apiKey: string, baseURL?: string): XaiProvider {
  const effectiveURL = baseURL || process.env.GROK_BASE_URL || DEFAULT_BASE_URL;

  if (effectiveURL !== DEFAULT_BASE_URL) {
    // Custom endpoint (OpenRouter, etc.): use OpenAI-compatible provider
    // which handles streaming tool call deltas correctly.
    const openai = createOpenAI({
      apiKey,
      baseURL: effectiveURL,
      compatibility: "compatible",
    });
    _openaiProvider = openai;
    return openai as unknown as XaiProvider;
  }

  _openaiProvider = null;
  return createXai({
    apiKey,
    baseURL: effectiveURL,
  });
}

export function resolveModelRuntime(provider: XaiProvider, requestedModelId: string): ResolvedModelRuntime {
  const customURL = isCustomBaseURL();
  // Skip model normalization for custom endpoints: pass the model ID
  // through as-is so provider-prefixed names like "x-ai/grok-4.1-fast"
  // reach the API unchanged.
  const modelId = customURL ? requestedModelId : normalizeModelId(requestedModelId);
  const modelInfo = customURL ? undefined : getModelInfo(modelId);
  const reasoningEffort = customURL
    ? undefined
    : getEffectiveReasoningEffort(modelId, getReasoningEffortForModel(modelId));

  let model: GrokRuntimeModel;
  if (customURL) {
    // Force Chat Completions API for custom endpoints (OpenRouter, etc.).
    // @ai-sdk/openai v3 defaults to the Responses API which most
    // third-party endpoints don't support.
    model = (_openaiProvider as OpenAIProvider).chat(modelId);
  } else if (modelInfo?.responsesOnly) {
    model = provider.responses(modelId);
  } else {
    model = provider(modelId);
  }

  return {
    model,
    modelId,
    modelInfo,
    providerOptions: reasoningEffort
      ? {
          xai: {
            reasoningEffort,
          },
        }
      : undefined,
  };
}

export async function generateTitle(provider: XaiProvider, userMessage: string): Promise<GeneratedTitle> {
  const runtime = resolveModelRuntime(provider, DEFAULT_TITLE_MODEL);
  try {
    const { text, usage } = await generateText({
      model: runtime.model,
      temperature: 0.5,
      ...(runtime.modelInfo?.supportsMaxOutputTokens === false ? {} : { maxOutputTokens: 60 }),
      ...(runtime.providerOptions ? { providerOptions: runtime.providerOptions } : {}),
      system: [
        "You are a title generator. Output ONLY a short title. Nothing else.",
        "Rules:",
        "- Single line, ≤50 characters",
        "- Use the same language as the user message",
        "- Focus on the main topic or intent",
        "- Keep technical terms, filenames, numbers exact",
        "- Remove filler words (the, this, my, a, an)",
        "- Never use tools or explain anything",
        "- If the message is a greeting, output something like 'Quick chat'",
      ].join("\n"),
      prompt: userMessage,
    });
    return {
      title: text?.trim().replace(/^["']|["']$/g, "") || "New session",
      modelId: runtime.modelId,
      usage,
    };
  } catch {
    return { title: "New session", modelId: runtime.modelId };
  }
}
