package adapter

import (
	"errors"
	"testing"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
)

func TestConvertOpenRouterStreamToAnthropicStream_BasicConversion(t *testing.T) {
	tests := []struct {
		name      string
		chunks    []*openrouter.ChatCompletionChunk
		options   []ConvertStreamOption
		wantCount int
		validate  func(events []anthropic.Event) bool
	}{
		{
			name:      "empty stream",
			chunks:    []*openrouter.ChatCompletionChunk{},
			wantCount: 2, // message_delta + message_stop
			validate: func(events []anthropic.Event) bool {
				if len(events) != 2 {
					return false
				}
				_, okDelta := events[0].(*anthropic.EventMessageDelta)
				_, okStop := events[1].(*anthropic.EventMessageStop)
				return okDelta && okStop
			},
		},
		{
			name: "single chunk with basic content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Content: "Hello world",
							},
						},
					},
				},
			},
			wantCount: 6, // message_start + content_block_start + content_block_delta + content_block_stop + message_delta + message_stop
			validate: func(events []anthropic.Event) bool {
				if len(events) != 6 {
					return false
				}

				// Check message_start
				start, ok := events[0].(*anthropic.EventMessageStart)
				if !ok || start.Message.ID != "chatcmpl-123" || start.Message.Model != "claude-3-5-sonnet-20241022" {
					return false
				}

				// Check content_block_start
				blockStart, ok := events[1].(*anthropic.EventContentBlockStart)
				if !ok || blockStart.Index != 0 || blockStart.ContentBlock.Type != anthropic.MessageContentTypeText {
					return false
				}

				// Check content_block_delta
				blockDelta, ok := events[2].(*anthropic.EventContentBlockDelta)
				if !ok || blockDelta.Index != 0 || blockDelta.Delta.Type != anthropic.MessageContentDeltaTypeTextDelta || blockDelta.Delta.Text != "Hello world" {
					return false
				}

				// Check content_block_stop
				blockStop, ok := events[3].(*anthropic.EventContentBlockStop)
				if !ok || blockStop.Index != 0 {
					return false
				}

				return true
			},
		},
		{
			name: "with input tokens option",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:      "chatcmpl-123",
					Model:   "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{},
				},
			},
			options:   []ConvertStreamOption{WithInputTokens(100)},
			wantCount: 3, // message_start + message_delta + message_stop
			validate: func(events []anthropic.Event) bool {
				start, ok := events[0].(*anthropic.EventMessageStart)
				return ok && start.Message.Usage.InputTokens == 100 && start.Message.Usage.OutputTokens == 1
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, nil)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream, tt.options...)

			var events []anthropic.Event
			for event, err := range anthropicStream {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				events = append(events, event)
			}

			if len(events) != tt.wantCount {
				t.Errorf("Expected %d events, got %d", tt.wantCount, len(events))
			}

			if !tt.validate(events) {
				t.Errorf("Event validation failed")
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_ContentTypes(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []*openrouter.ChatCompletionChunk
		validate func(events []anthropic.Event) bool
	}{
		{
			name: "text content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Content: "Hello",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Content: " world",
							},
						},
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				// Find content block deltas
				var textDeltas []*anthropic.EventContentBlockDelta
				for _, event := range events {
					if delta, ok := event.(*anthropic.EventContentBlockDelta); ok && delta.Delta.Type == anthropic.MessageContentDeltaTypeTextDelta {
						textDeltas = append(textDeltas, delta)
					}
				}

				return len(textDeltas) == 2 &&
					textDeltas[0].Delta.Text == "Hello" &&
					textDeltas[1].Delta.Text == " world"
			},
		},
		{
			name: "reasoning content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
									{
										Text:      "Let me think about this",
										Signature: "sig123",
									},
								},
							},
						},
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				// Find thinking deltas
				var thinkingDeltas []*anthropic.EventContentBlockDelta
				for _, event := range events {
					if delta, ok := event.(*anthropic.EventContentBlockDelta); ok && delta.Delta.Type == anthropic.MessageContentDeltaTypeThinkingDelta {
						thinkingDeltas = append(thinkingDeltas, delta)
					}
				}

				return len(thinkingDeltas) == 1 &&
					thinkingDeltas[0].Delta.Thinking == "Let me think about this"
			},
		},
		{
			name: "tool call content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								ToolCalls: []*openrouter.ChatCompletionToolCall{
									{
										ID: "tool_123",
										Function: &openrouter.ChatCompletionMessageToolCallFunction{
											Name:      "get_weather",
											Arguments: `{"location": "SF"}`,
										},
									},
								},
							},
						},
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				// Find content block start with tool use
				var toolStart *anthropic.EventContentBlockStart
				var inputDeltas []*anthropic.EventContentBlockDelta

				for _, event := range events {
					if start, ok := event.(*anthropic.EventContentBlockStart); ok && start.ContentBlock.Type == anthropic.MessageContentTypeToolUse {
						toolStart = start
					}
					if delta, ok := event.(*anthropic.EventContentBlockDelta); ok && delta.Delta.Type == anthropic.MessageContentDeltaTypeInputJSONDelta {
						inputDeltas = append(inputDeltas, delta)
					}
				}

				return toolStart != nil &&
					toolStart.ContentBlock.ID == "tool_123" &&
					toolStart.ContentBlock.Name == "get_weather" &&
					len(inputDeltas) == 1 &&
					inputDeltas[0].Delta.PartialJSON == `{"location": "SF"}`
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, nil)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream)

			var events []anthropic.Event
			for event, err := range anthropicStream {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				events = append(events, event)
			}

			if !tt.validate(events) {
				t.Errorf("Event validation failed")
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_ContentTypeTransitions(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []*openrouter.ChatCompletionChunk
		validate func(events []anthropic.Event) bool
	}{
		{
			name: "reasoning to text transition",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
									{Text: "Thinking..."},
								},
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Content: "My answer is",
							},
						},
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				// Should have: message_start, content_block_start (thinking), content_block_delta (thinking),
				// content_block_stop, content_block_start (text), content_block_delta (text), message_delta, message_stop

				var blockStops []*anthropic.EventContentBlockStop
				var blockStarts []*anthropic.EventContentBlockStart

				for _, event := range events {
					if stop, ok := event.(*anthropic.EventContentBlockStop); ok {
						blockStops = append(blockStops, stop)
					}
					if start, ok := event.(*anthropic.EventContentBlockStart); ok {
						blockStarts = append(blockStarts, start)
					}
				}

				return len(blockStops) == 2 && len(blockStarts) == 2 &&
					blockStarts[0].ContentBlock.Type == anthropic.MessageContentTypeThinking &&
					blockStops[0].Index == 0 &&
					blockStarts[1].ContentBlock.Type == anthropic.MessageContentTypeText &&
					blockStarts[1].Index == 1 &&
					blockStops[1].Index == 1
			},
		},
		{
			name: "text to tool call transition",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Content: "Let me check the weather",
							},
						},
					},
				},
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								ToolCalls: []*openrouter.ChatCompletionToolCall{
									{
										ID: "tool_123",
										Function: &openrouter.ChatCompletionMessageToolCallFunction{
											Name:      "get_weather",
											Arguments: "{}",
										},
									},
								},
							},
						},
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				var blockStops []*anthropic.EventContentBlockStop
				var blockStarts []*anthropic.EventContentBlockStart

				for _, event := range events {
					if stop, ok := event.(*anthropic.EventContentBlockStop); ok {
						blockStops = append(blockStops, stop)
					}
					if start, ok := event.(*anthropic.EventContentBlockStart); ok {
						blockStarts = append(blockStarts, start)
					}
				}

				return len(blockStops) == 2 && len(blockStarts) == 2 &&
					blockStarts[0].ContentBlock.Type == anthropic.MessageContentTypeText &&
					blockStops[0].Index == 0 &&
					blockStarts[1].ContentBlock.Type == anthropic.MessageContentTypeToolUse &&
					blockStarts[1].Index == 1 &&
					blockStops[1].Index == 1
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, nil)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream)

			var events []anthropic.Event
			for event, err := range anthropicStream {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				events = append(events, event)
			}

			if !tt.validate(events) {
				t.Errorf("Event validation failed")
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_FinishReasons(t *testing.T) {
	tests := []struct {
		name         string
		finishReason openrouter.ChatCompletionFinishReason
		expected     anthropic.StopReason
	}{
		{
			name:         "stop reason",
			finishReason: openrouter.ChatCompletionFinishReasonStop,
			expected:     anthropic.StopReasonEndTurn,
		},
		{
			name:         "length reason",
			finishReason: openrouter.ChatCompletionFinishReasonLength,
			expected:     anthropic.StopReasonMaxTokens,
		},
		{
			name:         "content filter reason",
			finishReason: openrouter.ChatCompletionFinishReasonContentFilter,
			expected:     anthropic.StopReasonRefusal,
		},
		{
			name:         "tool calls reason",
			finishReason: openrouter.ChatCompletionFinishReasonToolCalls,
			expected:     anthropic.StopReasonToolUse,
		},
		{
			name:         "unknown reason",
			finishReason: openrouter.ChatCompletionFinishReason("unknown"),
			expected:     anthropic.StopReasonPauseTurn,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							FinishReason: tt.finishReason,
						},
					},
				},
			}

			stream := createMockStream(chunks, nil)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream)

			var events []anthropic.Event
			for event, err := range anthropicStream {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				events = append(events, event)
			}

			// Find message delta with stop reason
			var messageDelta *anthropic.EventMessageDelta
			for _, event := range events {
				if delta, ok := event.(*anthropic.EventMessageDelta); ok {
					messageDelta = delta
					break
				}
			}

			if messageDelta == nil || messageDelta.Delta.StopReason == nil || *messageDelta.Delta.StopReason != tt.expected {
				t.Errorf("Expected stop reason %v, got %v", tt.expected, messageDelta.Delta.StopReason)
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_Usage(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []*openrouter.ChatCompletionChunk
		options  []ConvertStreamOption
		validate func(events []anthropic.Event) bool
	}{
		{
			name: "usage from chunk",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Usage: &openrouter.ChatCompletionUsage{
						PromptTokens:     50,
						CompletionTokens: 25,
						TotalTokens:      75,
					},
				},
			},
			validate: func(events []anthropic.Event) bool {
				// Find message delta
				var messageDelta *anthropic.EventMessageDelta
				for _, event := range events {
					if delta, ok := event.(*anthropic.EventMessageDelta); ok {
						messageDelta = delta
						break
					}
				}

				return messageDelta != nil &&
					messageDelta.Delta.Usage != nil &&
					messageDelta.Delta.Usage.InputTokens == 50 &&
					messageDelta.Delta.Usage.OutputTokens == 25
			},
		},
		{
			name: "initial usage from options",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
				},
			},
			options: []ConvertStreamOption{WithInputTokens(100)},
			validate: func(events []anthropic.Event) bool {
				// Find message start
				var messageStart *anthropic.EventMessageStart
				for _, event := range events {
					if start, ok := event.(*anthropic.EventMessageStart); ok {
						messageStart = start
						break
					}
				}

				return messageStart != nil &&
					messageStart.Message.Usage != nil &&
					messageStart.Message.Usage.InputTokens == 100 &&
					messageStart.Message.Usage.OutputTokens == 1
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, nil)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream, tt.options...)

			var events []anthropic.Event
			for event, err := range anthropicStream {
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
				events = append(events, event)
			}

			if !tt.validate(events) {
				t.Errorf("Usage validation failed")
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_ErrorHandling(t *testing.T) {
	tests := []struct {
		name        string
		streamError error
		validate    func(events []anthropic.Event, err error) bool
	}{
		{
			name:        "stream error propagates",
			streamError: errors.New("stream connection error"),
			validate: func(events []anthropic.Event, err error) bool {
				return err != nil && err.Error() == "stream connection error"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(nil, tt.streamError)
			anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream)

			var events []anthropic.Event
			var lastErr error

			for event, err := range anthropicStream {
				if err != nil {
					lastErr = err
					break
				}
				events = append(events, event)
			}

			if !tt.validate(events, lastErr) {
				t.Errorf("Error handling validation failed")
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_MultipleReasoningDetails(t *testing.T) {
	chunks := []*openrouter.ChatCompletionChunk{
		{
			ID:    "chatcmpl-123",
			Model: "claude-3-5-sonnet-20241022",
			Choices: []*openrouter.ChatCompletionChunkChoice{
				{
					Delta: &openrouter.ChatCompletionChunkChoiceDelta{
						ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
							{Text: "First thought", Signature: "sig123"},
							{Text: "Second thought", Signature: "sig456"},
						},
					},
				},
			},
		},
	}

	stream := createMockStream(chunks, nil)
	anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream)

	var thinkingDeltas []*anthropic.EventContentBlockDelta
	var signatureDeltas []*anthropic.EventContentBlockDelta

	for event, err := range anthropicStream {
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if delta, ok := event.(*anthropic.EventContentBlockDelta); ok {
			if delta.Delta.Type == anthropic.MessageContentDeltaTypeThinkingDelta {
				thinkingDeltas = append(thinkingDeltas, delta)
			} else if delta.Delta.Type == anthropic.MessageContentDeltaTypeSignatureDelta {
				signatureDeltas = append(signatureDeltas, delta)
			}
		}
	}

	if len(thinkingDeltas) != 2 {
		t.Errorf("Expected 2 thinking deltas, got %d", len(thinkingDeltas))
	}

	if len(signatureDeltas) != 2 {
		t.Errorf("Expected 2 signature deltas, got %d", len(signatureDeltas))
	}

	if thinkingDeltas[0].Delta.Thinking != "First thought" {
		t.Errorf("Expected 'First thought', got %s", thinkingDeltas[0].Delta.Thinking)
	}

	if signatureDeltas[0].Delta.Signature != "sig123" {
		t.Errorf("Expected 'sig123', got %s", signatureDeltas[0].Delta.Signature)
	}
}

func TestConvertOpenRouterStreamToAnthropicMessage_BasicConversion(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []*openrouter.ChatCompletionChunk
		validate func(message *anthropic.Message, err error) bool
	}{
		{
			name: "basic text content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Usage: &openrouter.ChatCompletionUsage{
						PromptTokens:     50,
						CompletionTokens: 25,
						TotalTokens:      75,
					},
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							FinishReason: openrouter.ChatCompletionFinishReasonStop,
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Role:    openrouter.ChatCompletionMessageRoleAssistant,
								Content: "Hello world",
							},
						},
					},
				},
			},
			validate: func(message *anthropic.Message, err error) bool {
				return err == nil &&
					message != nil &&
					message.ID == "chatcmpl-123" &&
					message.Type == anthropic.MessageTypeMessage &&
					message.Role == anthropic.MessageRoleAssistant &&
					message.Model == "claude-3-5-sonnet-20241022" &&
					message.Usage != nil &&
					message.Usage.InputTokens == 50 &&
					message.Usage.OutputTokens == 25 &&
					message.StopReason != nil &&
					*message.StopReason == anthropic.StopReasonEndTurn &&
					len(message.Content) == 1 &&
					message.Content[0].Type == anthropic.MessageContentTypeText &&
					message.Content[0].Text == "Hello world"
			},
		},
		{
			name: "with reasoning content",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-456",
					Model: "claude-3-5-sonnet-20241022",
					Usage: &openrouter.ChatCompletionUsage{
						PromptTokens:     100,
						CompletionTokens: 50,
						TotalTokens:      150,
					},
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							FinishReason: openrouter.ChatCompletionFinishReasonStop,
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Role:    openrouter.ChatCompletionMessageRoleAssistant,
								Content: "My answer",
								ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
									{
										Text:      "Let me think about this carefully",
										Signature: "sig123",
									},
								},
							},
						},
					},
				},
			},
			validate: func(message *anthropic.Message, err error) bool {
				return err == nil &&
					message != nil &&
					len(message.Content) == 2 &&
					message.Content[0].Type == anthropic.MessageContentTypeThinking &&
					message.Content[0].Thinking == "Let me think about this carefully" &&
					message.Content[0].Signature == "sig123" &&
					message.Content[1].Type == anthropic.MessageContentTypeText &&
					message.Content[1].Text == "My answer"
			},
		},
		{
			name: "with tool calls",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-789",
					Model: "claude-3-5-sonnet-20241022",
					Usage: &openrouter.ChatCompletionUsage{
						PromptTokens:     75,
						CompletionTokens: 30,
						TotalTokens:      105,
					},
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							FinishReason: openrouter.ChatCompletionFinishReasonToolCalls,
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Role:    openrouter.ChatCompletionMessageRoleAssistant,
								Content: "I'll check the weather for you",
								ToolCalls: []*openrouter.ChatCompletionToolCall{
									{
										ID: "tool_456",
										Function: &openrouter.ChatCompletionMessageToolCallFunction{
											Name:      "get_weather",
											Arguments: `{"location": "San Francisco"}`,
										},
									},
								},
							},
						},
					},
				},
			},
			validate: func(message *anthropic.Message, err error) bool {
				return err == nil &&
					message != nil &&
					message.StopReason != nil &&
					*message.StopReason == anthropic.StopReasonToolUse &&
					len(message.Content) == 2 &&
					message.Content[0].Type == anthropic.MessageContentTypeText &&
					message.Content[0].Text == "I'll check the weather for you" &&
					message.Content[1].Type == anthropic.MessageContentTypeToolUse &&
					message.Content[1].ID == "tool_456" &&
					message.Content[1].Name == "get_weather" &&
					string(message.Content[1].Input) == `{"location": "San Francisco"}`
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, nil)
			message, err := ConvertOpenRouterStreamToAnthropicMessage(stream)

			if !tt.validate(message, err) {
				t.Errorf("Validation failed for test case: %s", tt.name)
				if message != nil {
					t.Logf("Message ID: %s, Role: %s, Model: %s", message.ID, message.Role, message.Model)
					t.Logf("Content length: %d", len(message.Content))
					if len(message.Content) > 0 {
						t.Logf("First content type: %s", message.Content[0].Type)
					}
				}
				if err != nil {
					t.Logf("Error: %v", err)
				}
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicMessage_ErrorHandling(t *testing.T) {
	tests := []struct {
		name        string
		chunks      []*openrouter.ChatCompletionChunk
		streamError error
		wantError   bool
		errorMsg    string
	}{
		{
			name:        "stream error propagates",
			chunks:      nil,
			streamError: errors.New("connection timeout"),
			wantError:   true,
			errorMsg:    "connection timeout",
		},
		{
			name: "no choices error",
			chunks: []*openrouter.ChatCompletionChunk{
				{
					ID:      "chatcmpl-empty",
					Model:   "claude-3-5-sonnet-20241022",
					Choices: []*openrouter.ChatCompletionChunkChoice{},
				},
			},
			wantError: true,
			errorMsg:  "no choices found",
		},
		{
			name:      "empty stream",
			chunks:    []*openrouter.ChatCompletionChunk{},
			wantError: true,
			errorMsg:  "no choices found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := createMockStream(tt.chunks, tt.streamError)
			message, err := ConvertOpenRouterStreamToAnthropicMessage(stream)

			if tt.wantError {
				if err == nil {
					t.Errorf("Expected error but got none")
				} else if err.Error() != tt.errorMsg {
					t.Errorf("Expected error message '%s', got '%s'", tt.errorMsg, err.Error())
				}
				if message != nil {
					t.Errorf("Expected nil message on error, got %v", message)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if message == nil {
					t.Errorf("Expected non-nil message")
				}
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicMessage_FinishReasons(t *testing.T) {
	tests := []struct {
		name               string
		finishReason       openrouter.ChatCompletionFinishReason
		nativeFinishReason string
		expectedStopReason anthropic.StopReason
	}{
		{
			name:               "native finish reason as fallback for unknown reason",
			finishReason:       openrouter.ChatCompletionFinishReason("unknown_reason"),
			nativeFinishReason: "custom_stop",
			expectedStopReason: anthropic.StopReason("custom_stop"),
		},
		{
			name:               "standard stop reason",
			finishReason:       openrouter.ChatCompletionFinishReasonStop,
			expectedStopReason: anthropic.StopReasonEndTurn,
		},
		{
			name:               "length reason",
			finishReason:       openrouter.ChatCompletionFinishReasonLength,
			expectedStopReason: anthropic.StopReasonMaxTokens,
		},
		{
			name:               "content filter reason",
			finishReason:       openrouter.ChatCompletionFinishReasonContentFilter,
			expectedStopReason: anthropic.StopReasonRefusal,
		},
		{
			name:               "tool calls reason",
			finishReason:       openrouter.ChatCompletionFinishReasonToolCalls,
			expectedStopReason: anthropic.StopReasonToolUse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := []*openrouter.ChatCompletionChunk{
				{
					ID:    "chatcmpl-123",
					Model: "claude-3-5-sonnet-20241022",
					Usage: &openrouter.ChatCompletionUsage{
						PromptTokens:     10,
						CompletionTokens: 5,
						TotalTokens:      15,
					},
					Choices: []*openrouter.ChatCompletionChunkChoice{
						{
							FinishReason:       tt.finishReason,
							NativeFinishReason: tt.nativeFinishReason,
							Delta: &openrouter.ChatCompletionChunkChoiceDelta{
								Role:    openrouter.ChatCompletionMessageRoleAssistant,
								Content: "test",
							},
						},
					},
				},
			}

			stream := createMockStream(chunks, nil)
			message, err := ConvertOpenRouterStreamToAnthropicMessage(stream)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if message.StopReason == nil {
				t.Fatalf("Expected stop reason to be set")
			}

			if *message.StopReason != tt.expectedStopReason {
				t.Errorf("Expected stop reason %v, got %v", tt.expectedStopReason, *message.StopReason)
			}
		})
	}
}

func TestConvertOpenRouterStreamToAnthropicMessage_MultipleReasoningDetails(t *testing.T) {
	chunks := []*openrouter.ChatCompletionChunk{
		{
			ID:    "chatcmpl-multi",
			Model: "claude-3-5-sonnet-20241022",
			Usage: &openrouter.ChatCompletionUsage{
				PromptTokens:     60,
				CompletionTokens: 40,
				TotalTokens:      100,
			},
			Choices: []*openrouter.ChatCompletionChunkChoice{
				{
					FinishReason: openrouter.ChatCompletionFinishReasonStop,
					Delta: &openrouter.ChatCompletionChunkChoiceDelta{
						Role:    openrouter.ChatCompletionMessageRoleAssistant,
						Content: "Final answer",
						ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
							{
								Text:      "First reasoning step",
								Signature: "sig123",
								Index:     0,
							},
							{
								Text:      "Second reasoning step",
								Signature: "sig456",
								Index:     1,
							},
						},
					},
				},
			},
		},
	}

	stream := createMockStream(chunks, nil)
	message, err := ConvertOpenRouterStreamToAnthropicMessage(stream)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Should have 3 content blocks: 2 thinking + 1 text
	if len(message.Content) != 3 {
		t.Errorf("Expected 3 content blocks, got %d", len(message.Content))
		return
	}

	// Check first thinking block
	if message.Content[0].Type != anthropic.MessageContentTypeThinking ||
		message.Content[0].Thinking != "First reasoning step" ||
		message.Content[0].Signature != "sig123" {
		t.Errorf("First thinking block validation failed")
	}

	// Check second thinking block
	if message.Content[1].Type != anthropic.MessageContentTypeThinking ||
		message.Content[1].Thinking != "Second reasoning step" ||
		message.Content[1].Signature != "sig456" {
		t.Errorf("Second thinking block validation failed")
	}

	// Check text block
	if message.Content[2].Type != anthropic.MessageContentTypeText ||
		message.Content[2].Text != "Final answer" {
		t.Errorf("Text block validation failed")
	}
}

func TestConvertOpenRouterStreamToAnthropicMessage_MultipleToolCalls(t *testing.T) {
	chunks := []*openrouter.ChatCompletionChunk{
		{
			ID:    "chatcmpl-tools",
			Model: "claude-3-5-sonnet-20241022",
			Usage: &openrouter.ChatCompletionUsage{
				PromptTokens:     80,
				CompletionTokens: 60,
				TotalTokens:      140,
			},
			Choices: []*openrouter.ChatCompletionChunkChoice{
				{
					FinishReason: openrouter.ChatCompletionFinishReasonToolCalls,
					Delta: &openrouter.ChatCompletionChunkChoiceDelta{
						Role:    openrouter.ChatCompletionMessageRoleAssistant,
						Content: "I'll help you with both requests",
						ToolCalls: []*openrouter.ChatCompletionToolCall{
							{
								Index: 0,
								ID:    "tool_1",
								Function: &openrouter.ChatCompletionMessageToolCallFunction{
									Name:      "get_weather",
									Arguments: `{"location": "NYC"}`,
								},
							},
							{
								Index: 1,
								ID:    "tool_2",
								Function: &openrouter.ChatCompletionMessageToolCallFunction{
									Name:      "get_time",
									Arguments: `{"timezone": "EST"}`,
								},
							},
						},
					},
				},
			},
		},
	}

	stream := createMockStream(chunks, nil)
	message, err := ConvertOpenRouterStreamToAnthropicMessage(stream)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Should have 3 content blocks: 1 text + 2 tool_use
	if len(message.Content) != 3 {
		t.Errorf("Expected 3 content blocks, got %d", len(message.Content))
		return
	}

	// Check text block
	if message.Content[0].Type != anthropic.MessageContentTypeText ||
		message.Content[0].Text != "I'll help you with both requests" {
		t.Errorf("Text block validation failed")
	}

	// Check first tool call
	if message.Content[1].Type != anthropic.MessageContentTypeToolUse ||
		message.Content[1].ID != "tool_1" ||
		message.Content[1].Name != "get_weather" ||
		string(message.Content[1].Input) != `{"location": "NYC"}` {
		t.Errorf("First tool call validation failed")
	}

	// Check second tool call
	if message.Content[2].Type != anthropic.MessageContentTypeToolUse ||
		message.Content[2].ID != "tool_2" ||
		message.Content[2].Name != "get_time" ||
		string(message.Content[2].Input) != `{"timezone": "EST"}` {
		t.Errorf("Second tool call validation failed")
	}
}

// Helper function to create a mock OpenRouter stream
func createMockStream(chunks []*openrouter.ChatCompletionChunk, err error) openrouter.ChatCompletionStream {
	return func(yield func(*openrouter.ChatCompletionChunk, error) bool) {
		if err != nil {
			yield(nil, err)
			return
		}

		for _, chunk := range chunks {
			if !yield(chunk, nil) {
				return
			}
		}
	}
}
