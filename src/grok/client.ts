import { createOpenAI, type OpenAIProvider } from "@ai-sdk/openai";
import type { LanguageModelV3CallOptions, LanguageModelV3Message, LanguageModelV3Middleware } from "@ai-sdk/provider";
import { createXai } from "@ai-sdk/xai";
import { generateText, wrapLanguageModel } from "ai";
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

// ---------------------------------------------------------------------------
// OpenRouter adaptation: reverse-map OpenRouter model IDs to native IDs so
// that modelInfo (context window, capabilities) is available even when using
// a custom base URL.  Only dots-to-dashes in version numbers are converted;
// normalizeModelId() handles prefix stripping and alias lookup.
// ---------------------------------------------------------------------------
function resolveModelInfoForCustomURL(openRouterModelId: string): ModelInfo | undefined {
  const direct = getModelInfo(openRouterModelId);
  if (direct) return direct;

  const bare = openRouterModelId.replace(/^(x-ai|xai)\//i, "");
  // OpenRouter uses dots in versions (4.1) where native uses dashes (4-1).
  const dashed = bare.replace(/(\d+)\.(\d+)/g, "$1-$2");
  return getModelInfo(dashed) || getModelInfo(bare);
}

// ---------------------------------------------------------------------------
// OpenRouter adaptation: middleware that extracts images from MCP tool results
// and re-injects them as user messages so the model receives proper vision
// content instead of JSON-stringified base64 text.
//
// xAI/OpenRouter counts data-URL base64 as text tokens, so a single 1080p
// screenshot can cost 100-300K tokens per step if left in conversation
// history.  To avoid blowing the context window:
//   1. ALWAYS replace image data in tool results with a text placeholder
//      (prevents JSON-stringify of base64 in tool messages).
//   2. Only inject the actual image as a user message for the LATEST
//      un-responded-to tool results (the model sees it once).
//   3. For older tool results the model has already responded to, just
//      keep the "[screenshot]" placeholder — the model's earlier response
//      already captured what it saw.
// ---------------------------------------------------------------------------
function injectImagesFromToolResults(prompt: LanguageModelV3Message[]): LanguageModelV3Message[] {
  // Find the index of the last assistant message — tool results after it
  // are "pending" (model hasn't responded yet).
  let lastAssistantIdx = -1;
  for (let i = prompt.length - 1; i >= 0; i--) {
    if (prompt[i].role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }

  const out: LanguageModelV3Message[] = [];
  let modified = false;

  for (let i = 0; i < prompt.length; i++) {
    const msg = prompt[i];
    if (msg.role !== "tool") {
      out.push(msg);
      continue;
    }

    const isPending = i > lastAssistantIdx;
    const imageParts: Array<{ data: string; mediaType: string }> = [];
    const rewrittenContent = msg.content.map((part) => {
      if (part.type !== "tool-result") return part;
      const output = part.output;
      if (!output || output.type !== "content" || !Array.isArray(output.value)) return part;

      const hasImage = output.value.some((v: { type: string }) => v.type === "image-data" || v.type === "file-data");
      if (!hasImage) return part;

      modified = true;
      const keptValues: typeof output.value = [];
      for (const item of output.value) {
        if ((item.type === "image-data" || item.type === "file-data") && item.mediaType?.startsWith("image/")) {
          if (isPending) {
            imageParts.push({ data: item.data, mediaType: item.mediaType });
          }
          keptValues.push({ type: "text" as const, text: "[screenshot]" });
        } else {
          keptValues.push(item);
        }
      }
      return { ...part, output: { ...output, value: keptValues } };
    });

    out.push({ ...msg, content: rewrittenContent } as typeof msg);

    if (imageParts.length > 0) {
      out.push({
        role: "user" as const,
        content: [
          ...imageParts.map((img) => ({
            type: "file" as const,
            data: img.data,
            mediaType: img.mediaType,
          })),
          { type: "text" as const, text: "Above is the screenshot from the tool call." },
        ],
      });
    }
  }

  return modified ? out : prompt;
}

const imageInjectionMiddleware: LanguageModelV3Middleware = {
  specificationVersion: "v3",
  transformParams: async ({ params }: { params: LanguageModelV3CallOptions }) => ({
    ...params,
    prompt: injectImagesFromToolResults(params.prompt),
  }),
};

export function resolveModelRuntime(provider: XaiProvider, requestedModelId: string): ResolvedModelRuntime {
  const customURL = isCustomBaseURL();
  const modelId = customURL ? requestedModelId : normalizeModelId(requestedModelId);

  // For custom endpoints, reverse-map the OpenRouter model ID to find native
  // modelInfo (enables context compaction, capability detection, etc.).
  const modelInfo = customURL ? resolveModelInfoForCustomURL(requestedModelId) : getModelInfo(modelId);
  const nativeId = modelInfo?.id;
  const reasoningEffort = customURL
    ? nativeId
      ? getEffectiveReasoningEffort(nativeId, getReasoningEffortForModel(nativeId))
      : undefined
    : getEffectiveReasoningEffort(modelId, getReasoningEffortForModel(modelId));

  let model: GrokRuntimeModel;
  if (customURL) {
    // Force Chat Completions API for custom endpoints (OpenRouter, etc.).
    // @ai-sdk/openai v3 defaults to the Responses API which most
    // third-party endpoints don't support.
    const baseModel = (_openaiProvider as OpenAIProvider).chat(modelId);
    // Wrap with image-injection middleware so MCP screenshot results reach the
    // model as proper vision content rather than JSON-stringified base64.
    model = wrapLanguageModel({
      model: baseModel,
      middleware: imageInjectionMiddleware,
    }) as unknown as GrokRuntimeModel;
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
