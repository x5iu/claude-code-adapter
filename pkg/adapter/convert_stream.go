package adapter

import (
	"encoding/json"
	"path/filepath"
	"sync"

	"github.com/samber/lo"
	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
)

func init() {
	viper.SetDefault(delimiter.ViperKey("options", "reasoning", "delimiter"), string(filepath.Separator))
	viper.SetDefault(delimiter.ViperKey("options", "context_window_resize_factor"), 1.0)
}

type ConvertStreamOptions struct {
	InputTokens                     int64
	OpenRouterProvider              *string
	OpenRouterChatCompletionBuilder *openrouter.ChatCompletionBuilder
}

type ConvertStreamOption func(*ConvertStreamOptions)

func WithInputTokens(inputTokens int64) ConvertStreamOption {
	return func(o *ConvertStreamOptions) {
		o.InputTokens = inputTokens
	}
}

func ExtractOpenRouterProvider(provider *string) ConvertStreamOption {
	return func(o *ConvertStreamOptions) {
		o.OpenRouterProvider = provider
	}
}

func ExtractOpenRouterChatCompletionBuilder(builder *openrouter.ChatCompletionBuilder) ConvertStreamOption {
	return func(o *ConvertStreamOptions) {
		o.OpenRouterChatCompletionBuilder = builder
	}
}

func ConvertOpenRouterStreamToAnthropicStream(
	stream openrouter.ChatCompletionStream,
	options ...ConvertStreamOption,
) anthropic.MessageStream {
	convertOptions := &ConvertStreamOptions{}
	for _, applyOption := range options {
		applyOption(convertOptions)
	}
	contextWindowResizeFactor := viper.GetFloat64(delimiter.ViperKey("options", "context_window_resize_factor"))
	return func(yield func(anthropic.Event, error) bool) {
		var (
			startOnce  sync.Once
			blockIndex int
			deltaType  anthropic.MessageContentDeltaType
			toolCallID string
			stopReason anthropic.StopReason
			usage      *anthropic.Usage
		)
		for chunk, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}
			startOnce.Do(func() {
				if convertOptions.OpenRouterProvider != nil && chunk.Provider != "" {
					*convertOptions.OpenRouterProvider = chunk.Provider
				}
				yield(&anthropic.EventMessageStart{
					Type: anthropic.EventTypeMessageStart,
					Message: &anthropic.Message{
						ID:    chunk.ID,
						Type:  anthropic.MessageTypeMessage,
						Role:  anthropic.MessageRoleAssistant,
						Model: chunk.Model,
						Usage: &anthropic.Usage{
							InputTokens:  int64(float64(convertOptions.InputTokens) * contextWindowResizeFactor),
							OutputTokens: 1,
						},
					},
				}, nil)
			})
			if convertOptions.OpenRouterChatCompletionBuilder != nil {
				convertOptions.OpenRouterChatCompletionBuilder.Add(chunk)
			}
			if chunk.Usage != nil {
				usage = &anthropic.Usage{
					InputTokens:  int64(float64(chunk.Usage.PromptTokens) * contextWindowResizeFactor),
					OutputTokens: int64(float64(chunk.Usage.CompletionTokens) * contextWindowResizeFactor),
				}
				if promptTokensDetails := chunk.Usage.PromptTokensDetails; promptTokensDetails != nil {
					usage.CacheReadInputTokens = int64(float64(promptTokensDetails.CachedTokens) * contextWindowResizeFactor)
				}
			}
			if choices := chunk.Choices; len(choices) > 0 {
				choice := choices[0]
				if finishReason := choice.FinishReason; finishReason != "" {
					// google-gemini-v1 will give finish_reason multiple times, so we should ensure stopReason is only set once.
					if stopReason == "" {
						stopReason = ConvertOpenRouterFinishReasonToAnthropicStopReason(finishReason, choice.NativeFinishReason)
					}
				}
				if delta := choice.Delta; delta != nil {
					if reasoningDetails := delta.ReasoningDetails; len(reasoningDetails) > 0 {
						if deltaType != anthropic.MessageContentDeltaTypeThinkingDelta {
							if deltaType != "" {
								blockStop := &anthropic.EventContentBlockStop{
									Type:  anthropic.EventTypeContentBlockStop,
									Index: blockIndex,
								}
								if !yield(blockStop, nil) {
									return
								}
								blockIndex++
							}
							deltaType = anthropic.MessageContentDeltaTypeThinkingDelta
							blockStart := &anthropic.EventContentBlockStart{
								Type:  anthropic.EventTypeContentBlockStart,
								Index: blockIndex,
								ContentBlock: &anthropic.MessageContent{
									Type: anthropic.MessageContentTypeThinking,
								},
							}
							if !yield(blockStart, nil) {
								return
							}
						}
						for _, reasoningDetail := range reasoningDetails {
							switch reasoningDetail.Type {
							case openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText:
								// Claude Code will crash when it encounters an empty thinking field, so we need to ensure that
								// every event contains valid thinking characters.
								//
								// And the bad news is that OpenRouter tends to output empty thinking deltas.
								if reasoningDetailText := reasoningDetail.Text; reasoningDetailText != "" {
									blockDelta := &anthropic.EventContentBlockDelta{
										Type:  anthropic.EventTypeContentBlockDelta,
										Index: blockIndex,
										Delta: &anthropic.MessageContentDelta{
											Type:     anthropic.MessageContentDeltaTypeThinkingDelta,
											Thinking: reasoningDetailText,
										},
									}
									if !yield(blockDelta, nil) {
										return
									}
								}
								if reasoningDetail.Signature != "" {
									blockDelta := &anthropic.EventContentBlockDelta{
										Type:  anthropic.EventTypeContentBlockDelta,
										Index: blockIndex,
										Delta: &anthropic.MessageContentDelta{
											Type:      anthropic.MessageContentDeltaTypeSignatureDelta,
											Signature: reasoningDetail.Signature,
										},
									}
									if !yield(blockDelta, nil) {
										return
									}
								}
							case openrouter.ChatCompletionMessageReasoningDetailTypeSummary:
								if reasoningDetailSummary := reasoningDetail.Summary; reasoningDetailSummary != "" {
									blockDelta := &anthropic.EventContentBlockDelta{
										Type:  anthropic.EventTypeContentBlockDelta,
										Index: blockIndex,
										Delta: &anthropic.MessageContentDelta{
											Type:     anthropic.MessageContentDeltaTypeThinkingDelta,
											Thinking: reasoningDetailSummary,
										},
									}
									if !yield(blockDelta, nil) {
										return
									}
								}
							case openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted:
								if reasoningDetailData := reasoningDetail.Data; reasoningDetailData != "" {
									var signature string
									if reasoningDetailID := reasoningDetail.ID; reasoningDetailID != "" {
										signature = reasoningDetailID + viper.GetString(delimiter.ViperKey("options", "reasoning", "delimiter")) + reasoningDetailData
									} else {
										signature = reasoningDetailData
									}
									blockDelta := &anthropic.EventContentBlockDelta{
										Type:  anthropic.EventTypeContentBlockDelta,
										Index: blockIndex,
										Delta: &anthropic.MessageContentDelta{
											Type:      anthropic.MessageContentDeltaTypeSignatureDelta,
											Signature: signature,
										},
									}
									if !yield(blockDelta, nil) {
										return
									}
								}
							}
						}
					}
					if content := delta.Content; content != "" {
						if deltaType != anthropic.MessageContentDeltaTypeTextDelta {
							if deltaType != "" {
								blockStop := &anthropic.EventContentBlockStop{
									Type:  anthropic.EventTypeContentBlockStop,
									Index: blockIndex,
								}
								if !yield(blockStop, nil) {
									return
								}
								blockIndex++
							}
							deltaType = anthropic.MessageContentDeltaTypeTextDelta
							blockStart := &anthropic.EventContentBlockStart{
								Type:  anthropic.EventTypeContentBlockStart,
								Index: blockIndex,
								ContentBlock: &anthropic.MessageContent{
									Type: anthropic.MessageContentTypeText,
								},
							}
							if !yield(blockStart, nil) {
								return
							}
						}
						blockDelta := &anthropic.EventContentBlockDelta{
							Type:  anthropic.EventTypeContentBlockDelta,
							Index: blockIndex,
							Delta: &anthropic.MessageContentDelta{
								Type: anthropic.MessageContentDeltaTypeTextDelta,
								Text: content,
							},
						}
						if !yield(blockDelta, nil) {
							return
						}
					}
					if toolCalls := delta.ToolCalls; len(toolCalls) > 0 {
						if toolCall := toolCalls[0]; toolCall.Function != nil {
							if deltaType != anthropic.MessageContentDeltaTypeInputJSONDelta || (toolCall.ID != "" && toolCall.ID != toolCallID) {
								if deltaType != "" {
									blockStop := &anthropic.EventContentBlockStop{
										Type:  anthropic.EventTypeContentBlockStop,
										Index: blockIndex,
									}
									if !yield(blockStop, nil) {
										return
									}
									blockIndex++
								}
								deltaType = anthropic.MessageContentDeltaTypeInputJSONDelta
								toolCallID = toolCall.ID
								blockStart := &anthropic.EventContentBlockStart{
									Type:  anthropic.EventTypeContentBlockStart,
									Index: blockIndex,
									ContentBlock: &anthropic.MessageContent{
										Type:  anthropic.MessageContentTypeToolUse,
										ID:    toolCall.ID,
										Name:  toolCall.Function.Name,
										Input: json.RawMessage("{}"),
									},
								}
								if !yield(blockStart, nil) {
									return
								}
							}
							// Claude Code will stop outputting when it encounters an empty partial_json field, so we need to ensure that
							// every event contains valid partial_json characters.
							//
							// And the bad news is that OpenRouter tends to output empty function arguments.
							if arguments := toolCall.Function.Arguments; arguments != "" {
								blockDelta := &anthropic.EventContentBlockDelta{
									Type:  anthropic.EventTypeContentBlockDelta,
									Index: blockIndex,
									Delta: &anthropic.MessageContentDelta{
										Type:        anthropic.MessageContentDeltaTypeInputJSONDelta,
										PartialJSON: toolCall.Function.Arguments,
									},
								}
								if !yield(blockDelta, nil) {
									return
								}
							}
						}
					}
				}
			}
		}
		if deltaType != "" {
			blockEnd := &anthropic.EventContentBlockStop{
				Type:  anthropic.EventTypeContentBlockStop,
				Index: blockIndex,
			}
			if !yield(blockEnd, nil) {
				return
			}
		}
		delta := &anthropic.Message{}
		if stopReason != "" {
			delta.StopReason = lo.ToPtr(stopReason)
		}
		if usage != nil {
			delta.Usage = usage
		}
		messageDelta := &anthropic.EventMessageDelta{
			Type:  anthropic.EventTypeMessageDelta,
			Delta: delta,
			Usage: usage,
		}
		if !yield(messageDelta, nil) {
			return
		}
		yield(&anthropic.EventMessageStop{Type: anthropic.EventTypeMessageStop}, nil)
	}
}

func ConvertOpenRouterFinishReasonToAnthropicStopReason(
	finishReason openrouter.ChatCompletionFinishReason,
	nativeFinishReason string,
) (stopReason anthropic.StopReason) {
	switch finishReason {
	case openrouter.ChatCompletionFinishReasonStop:
		stopReason = anthropic.StopReasonEndTurn
	case openrouter.ChatCompletionFinishReasonLength:
		stopReason = anthropic.StopReasonMaxTokens
	case openrouter.ChatCompletionFinishReasonContentFilter:
		stopReason = anthropic.StopReasonRefusal
	case openrouter.ChatCompletionFinishReasonToolCalls:
		stopReason = anthropic.StopReasonToolUse
	default:
		if nativeFinishReason != "" {
			stopReason = anthropic.StopReason(nativeFinishReason)
		} else {
			stopReason = anthropic.StopReasonPauseTurn
		}
	}
	return stopReason
}
