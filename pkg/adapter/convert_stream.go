package adapter

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/samber/lo"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openai"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/profile"
)

type ConvertStreamOptions struct {
	InputTokens                     int64
	OpenRouterProvider              *string
	OpenRouterChatCompletionBuilder *openrouter.ChatCompletionBuilder
	OpenAIResponseBuilder           *openai.ResponseBuilder
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

func ExtractOpenAIResponseBuilder(builder *openai.ResponseBuilder) ConvertStreamOption {
	return func(o *ConvertStreamOptions) {
		o.OpenAIResponseBuilder = builder
	}
}

func ConvertOpenRouterStreamToAnthropicStream(
	ctx context.Context,
	stream openrouter.ChatCompletionStream,
	options ...ConvertStreamOption,
) anthropic.MessageStream {
	prof, _ := profile.FromContext(ctx)
	convertOptions := &ConvertStreamOptions{}
	for _, applyOption := range options {
		applyOption(convertOptions)
	}
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
							InputTokens:  convertOptions.InputTokens,
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
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
				}
				if promptTokensDetails := chunk.Usage.PromptTokensDetails; promptTokensDetails != nil {
					usage.CacheReadInputTokens = promptTokensDetails.CachedTokens
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
					if reasoningDetails := delta.ReasoningDetails; reasoningDetailsContainsReasoningTypes(reasoningDetails,
						openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						openrouter.ChatCompletionMessageReasoningDetailTypeSummary,
					) {
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
					if reasoningDetails := delta.ReasoningDetails; reasoningDetailsContainsReasoningTypes(reasoningDetails,
						openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted,
					) {
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
							case openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted:
								if reasoningDetailData := reasoningDetail.Data; reasoningDetailData != "" {
									var signature string
									if reasoningDetailID := reasoningDetail.ID; reasoningDetailID != "" {
										signature = reasoningDetailID + prof.Options.GetReasoningDelimiter() + reasoningDetailData
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

func reasoningDetailsContainsReasoningTypes(
	details []*openrouter.ChatCompletionMessageReasoningDetail,
	reasoningTypes ...openrouter.ChatCompletionMessageReasoningDetailType,
) bool {
	for _, detail := range details {
		if lo.Contains(reasoningTypes, detail.Type) {
			return true
		}
	}
	return false
}

func ConvertOpenAIStreamToAnthropicStream(
	ctx context.Context,
	stream openai.ResponseStream,
	options ...ConvertStreamOption,
) anthropic.MessageStream {
	prof, _ := profile.FromContext(ctx)
	convertOptions := &ConvertStreamOptions{}
	for _, applyOption := range options {
		applyOption(convertOptions)
	}
	contextWindowResizeFactor := prof.Options.GetContextWindowResizeFactor()
	return func(yield func(anthropic.Event, error) bool) {
		var (
			startOnce  sync.Once
			responseID string
			model      string
			blockIndex int
			deltaType  anthropic.MessageContentDeltaType
			toolCallID string
			stopReason anthropic.StopReason
			usage      *anthropic.Usage
		)
		for event, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}
			if convertOptions.OpenAIResponseBuilder != nil {
				convertOptions.OpenAIResponseBuilder.Add(event)
			}
			switch e := event.(type) {
			case *openai.ResponseCreatedEvent:
				responseID = e.Response.ID
				model = e.Response.Model
				startOnce.Do(func() {
					yield(&anthropic.EventMessageStart{
						Type: anthropic.EventTypeMessageStart,
						Message: &anthropic.Message{
							ID:    responseID,
							Type:  anthropic.MessageTypeMessage,
							Role:  anthropic.MessageRoleAssistant,
							Model: model,
							Usage: &anthropic.Usage{
								InputTokens:  int64(float64(convertOptions.InputTokens) * contextWindowResizeFactor),
								OutputTokens: 1,
							},
						},
					}, nil)
				})
			case *openai.ResponseInProgressEvent:
				// No action needed
			case *openai.ResponseOutputItemAddedEvent:
				// Handle output item based on its type
				if e.Item != nil {
					if msg := e.Item.Message; msg != nil {
						// Message item - we'll handle content deltas separately
					} else if fc := e.Item.FunctionCall; fc != nil {
						// Function call started
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeInputJSONDelta
						toolCallID = fc.CallID
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type:  anthropic.MessageContentTypeToolUse,
								ID:    fc.CallID,
								Name:  fc.Name,
								Input: json.RawMessage("{}"),
							},
						}, nil) {
							return
						}
					} else if r := e.Item.Reasoning; r != nil {
						// Reasoning item started
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeThinkingDelta
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type: anthropic.MessageContentTypeThinking,
							},
						}, nil) {
							return
						}
					}
				}
			case *openai.ResponseOutputItemDoneEvent:
				// Output item completed - no additional action needed
			case *openai.ResponseContentPartAddedEvent:
				// Content part added - we'll handle deltas
				if e.Part != nil {
					// Part is ResponseMessageContentText, check its type
					if e.Part.Type == openai.ResponseMessageContentTypeInputText ||
						e.Part.Type == openai.ResponseMessageContentTypeOutputText {
						// Text content started
						if deltaType != anthropic.MessageContentDeltaTypeTextDelta {
							if deltaType != "" {
								if !yield(&anthropic.EventContentBlockStop{
									Type:  anthropic.EventTypeContentBlockStop,
									Index: blockIndex,
								}, nil) {
									return
								}
								blockIndex++
							}
							deltaType = anthropic.MessageContentDeltaTypeTextDelta
							if !yield(&anthropic.EventContentBlockStart{
								Type:  anthropic.EventTypeContentBlockStart,
								Index: blockIndex,
								ContentBlock: &anthropic.MessageContent{
									Type: anthropic.MessageContentTypeText,
								},
							}, nil) {
								return
							}
						}
					}
				}
			case *openai.ResponseContentPartDoneEvent:
				// Content part done - no additional action needed
			case *openai.ResponseTextDeltaEvent:
				// Text delta
				if e.Delta != "" {
					if deltaType != anthropic.MessageContentDeltaTypeTextDelta {
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeTextDelta
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type: anthropic.MessageContentTypeText,
							},
						}, nil) {
							return
						}
					}
					if !yield(&anthropic.EventContentBlockDelta{
						Type:  anthropic.EventTypeContentBlockDelta,
						Index: blockIndex,
						Delta: &anthropic.MessageContentDelta{
							Type: anthropic.MessageContentDeltaTypeTextDelta,
							Text: e.Delta,
						},
					}, nil) {
						return
					}
				}
			case *openai.ResponseTextDoneEvent:
				// Text done - no additional action needed
			case *openai.ResponseReasoningTextDeltaEvent:
				// Reasoning text delta
				if e.Delta != "" {
					if deltaType != anthropic.MessageContentDeltaTypeThinkingDelta {
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeThinkingDelta
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type: anthropic.MessageContentTypeThinking,
							},
						}, nil) {
							return
						}
					}
					if !yield(&anthropic.EventContentBlockDelta{
						Type:  anthropic.EventTypeContentBlockDelta,
						Index: blockIndex,
						Delta: &anthropic.MessageContentDelta{
							Type:     anthropic.MessageContentDeltaTypeThinkingDelta,
							Thinking: e.Delta,
						},
					}, nil) {
						return
					}
				}
			case *openai.ResponseReasoningTextDoneEvent:
				// Reasoning text done - no additional action needed
			case *openai.ResponseReasoningSummaryTextDeltaEvent:
				// Reasoning summary text delta (treat as thinking)
				if e.Delta != "" {
					if deltaType != anthropic.MessageContentDeltaTypeThinkingDelta {
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeThinkingDelta
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type: anthropic.MessageContentTypeThinking,
							},
						}, nil) {
							return
						}
					}
					if !yield(&anthropic.EventContentBlockDelta{
						Type:  anthropic.EventTypeContentBlockDelta,
						Index: blockIndex,
						Delta: &anthropic.MessageContentDelta{
							Type:     anthropic.MessageContentDeltaTypeThinkingDelta,
							Thinking: e.Delta,
						},
					}, nil) {
						return
					}
				}
			case *openai.ResponseReasoningSummaryTextDoneEvent:
				// Reasoning summary done - no additional action needed
			case *openai.ResponseFunctionCallArgumentsDeltaEvent:
				// Function call arguments delta
				if e.Delta != "" {
					if deltaType != anthropic.MessageContentDeltaTypeInputJSONDelta || (e.ItemID != "" && e.ItemID != toolCallID) {
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeInputJSONDelta
						toolCallID = e.ItemID
						// Note: We don't have the function name here, it should have been set in OutputItemAdded
					}
					if !yield(&anthropic.EventContentBlockDelta{
						Type:  anthropic.EventTypeContentBlockDelta,
						Index: blockIndex,
						Delta: &anthropic.MessageContentDelta{
							Type:        anthropic.MessageContentDeltaTypeInputJSONDelta,
							PartialJSON: e.Delta,
						},
					}, nil) {
						return
					}
				}
			case *openai.ResponseFunctionCallArgumentsDoneEvent:
				// Function call arguments done - no additional action needed
			case *openai.ResponseCompletedEvent:
				// Response completed
				stopReason = convertOpenAIStatusToAnthropicStopReason(e.Response.Status, e.Response.IncompleteDetails)
				if e.Response.Usage != nil {
					usage = &anthropic.Usage{
						InputTokens:  int64(float64(e.Response.Usage.InputTokens) * contextWindowResizeFactor),
						OutputTokens: int64(float64(e.Response.Usage.OutputTokens) * contextWindowResizeFactor),
					}
					if details := e.Response.Usage.InputTokensDetails; details != nil {
						usage.CacheReadInputTokens = int64(float64(details.CachedTokens) * contextWindowResizeFactor)
					}
				}
			case *openai.ResponseFailedEvent:
				// Response failed
				stopReason = anthropic.StopReasonRefusal
			case *openai.ResponseIncompleteEvent:
				// Response incomplete
				if e.Response.IncompleteDetails != nil {
					switch e.Response.IncompleteDetails.Reason {
					case openai.ResponseIncompleteReasonMaxOutputTokens:
						stopReason = anthropic.StopReasonMaxTokens
					case openai.ResponseIncompleteReasonContentFilter:
						stopReason = anthropic.StopReasonRefusal
					default:
						stopReason = anthropic.StopReasonPauseTurn
					}
				} else {
					stopReason = anthropic.StopReasonPauseTurn
				}
			case *openai.ResponseRefusalDeltaEvent:
				// Refusal delta - treat as text
				if e.Delta != "" {
					if deltaType != anthropic.MessageContentDeltaTypeTextDelta {
						if deltaType != "" {
							if !yield(&anthropic.EventContentBlockStop{
								Type:  anthropic.EventTypeContentBlockStop,
								Index: blockIndex,
							}, nil) {
								return
							}
							blockIndex++
						}
						deltaType = anthropic.MessageContentDeltaTypeTextDelta
						if !yield(&anthropic.EventContentBlockStart{
							Type:  anthropic.EventTypeContentBlockStart,
							Index: blockIndex,
							ContentBlock: &anthropic.MessageContent{
								Type: anthropic.MessageContentTypeText,
							},
						}, nil) {
							return
						}
					}
					if !yield(&anthropic.EventContentBlockDelta{
						Type:  anthropic.EventTypeContentBlockDelta,
						Index: blockIndex,
						Delta: &anthropic.MessageContentDelta{
							Type: anthropic.MessageContentDeltaTypeTextDelta,
							Text: e.Delta,
						},
					}, nil) {
						return
					}
				}
			case *openai.ResponseErrorEvent:
				// Error event
				yield(nil, &openai.Error{
					Inner: struct {
						Message string `json:"message"`
						Type    string `json:"type"`
						Param   any    `json:"param,omitempty"`
						Code    string `json:"code"`
					}{
						Message: e.Message,
						Code:    string(e.Code),
					},
				})
				return
			}
		}
		// Close any open content block
		if deltaType != "" {
			if !yield(&anthropic.EventContentBlockStop{
				Type:  anthropic.EventTypeContentBlockStop,
				Index: blockIndex,
			}, nil) {
				return
			}
		}
		// Send message delta and stop
		delta := &anthropic.Message{}
		if stopReason != "" {
			delta.StopReason = lo.ToPtr(stopReason)
		} else {
			delta.StopReason = lo.ToPtr(anthropic.StopReasonEndTurn)
		}
		if usage != nil {
			delta.Usage = usage
		}
		if !yield(&anthropic.EventMessageDelta{
			Type:  anthropic.EventTypeMessageDelta,
			Delta: delta,
			Usage: usage,
		}, nil) {
			return
		}
		yield(&anthropic.EventMessageStop{Type: anthropic.EventTypeMessageStop}, nil)
	}
}

func convertOpenAIStatusToAnthropicStopReason(
	status openai.ResponseStatus,
	incompleteDetails *openai.ResponseIncompleteDetails,
) anthropic.StopReason {
	switch status {
	case openai.ResponseStatusCompleted:
		return anthropic.StopReasonEndTurn
	case openai.ResponseStatusIncomplete:
		if incompleteDetails != nil {
			switch incompleteDetails.Reason {
			case openai.ResponseIncompleteReasonMaxOutputTokens:
				return anthropic.StopReasonMaxTokens
			case openai.ResponseIncompleteReasonContentFilter:
				return anthropic.StopReasonRefusal
			}
		}
		return anthropic.StopReasonPauseTurn
	case openai.ResponseStatusFailed:
		return anthropic.StopReasonRefusal
	default:
		return anthropic.StopReasonEndTurn
	}
}
