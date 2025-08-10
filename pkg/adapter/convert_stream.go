package adapter

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/samber/lo"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
)

type ConvertStreamOptions struct {
	InputTokens        int64
	OpenRouterProvider *string
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

func ConvertOpenRouterStreamToAnthropicStream(
	stream openrouter.ChatCompletionStream,
	options ...ConvertStreamOption,
) anthropic.MessageStream {
	convertOptions := &ConvertStreamOptions{}
	for _, applyOption := range options {
		applyOption(convertOptions)
	}
	return func(yield func(anthropic.Event, error) bool) {
		var (
			startOnce  sync.Once
			blockIndex int
			deltaType  anthropic.MessageContentDeltaType
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
							InputTokens:  convertOptions.InputTokens,
							OutputTokens: 1,
						},
					},
				}, nil)
			})
			if chunk.Usage != nil {
				usage = &anthropic.Usage{
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
				}
			}
			if choices := chunk.Choices; len(choices) > 0 {
				choice := choices[0]
				if finishReason := choice.FinishReason; finishReason != "" {
					stopReason = ConvertOpenRouterFinishReasonToAnthropicStopReason(finishReason, choice.NativeFinishReason)
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
							if deltaType != anthropic.MessageContentDeltaTypeInputJSONDelta {
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

func ConvertOpenRouterStreamToAnthropicMessage(
	stream openrouter.ChatCompletionStream,
) (*anthropic.Message, error) {
	builder := openrouter.NewChatCompletionBuilder()
	for chunk, err := range stream {
		if err != nil {
			return nil, err
		}
		builder.Add(chunk)
	}
	src := builder.Build()
	dst := &anthropic.Message{
		ID:    src.ID,
		Type:  anthropic.MessageTypeMessage,
		Role:  anthropic.MessageRoleAssistant,
		Model: src.Model,
		Usage: &anthropic.Usage{
			InputTokens:  src.GetPromptTokens(),
			OutputTokens: src.GetCompletionTokens(),
		},
	}
	if len(src.Choices) == 0 {
		return nil, fmt.Errorf("no choices found")
	}
	dst.StopReason = lo.ToPtr(ConvertOpenRouterFinishReasonToAnthropicStopReason(
		src.Choices[0].FinishReason,
		src.Choices[0].NativeFinishReason,
	))
	srcMessage := src.Choices[0].Message
	for _, reasoningDetail := range srcMessage.ReasoningDetails {
		dst.Content = append(dst.Content, &anthropic.MessageContent{
			Type:      anthropic.MessageContentTypeThinking,
			Thinking:  reasoningDetail.Text,
			Signature: reasoningDetail.Signature,
		})
	}
	dst.Content = append(dst.Content, &anthropic.MessageContent{
		Type: anthropic.MessageContentTypeText,
		Text: srcMessage.Content.Text,
	})
	for _, toolCall := range srcMessage.ToolCalls {
		if function := toolCall.Function; function != nil {
			dst.Content = append(dst.Content, &anthropic.MessageContent{
				Type:  anthropic.MessageContentTypeToolUse,
				ID:    toolCall.ID,
				Name:  function.Name,
				Input: json.RawMessage(function.Arguments),
			})
		}
	}
	return dst, nil
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
