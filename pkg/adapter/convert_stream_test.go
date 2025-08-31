package adapter

import (
	"errors"
	"testing"

	"github.com/spf13/viper"

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
										Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
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
							{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "First thought", Signature: "sig123"},
							{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "Second thought", Signature: "sig456"},
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

func TestConvertOpenRouterStreamToAnthropicStream_ProviderExtraction(t *testing.T) {
	providerName := "google-vertex"
	var orProvider string
	chunks := []*openrouter.ChatCompletionChunk{{ID: "chatcmpl-x", Model: "m", Provider: providerName}}
	stream := createMockStream(chunks, nil)
	anthropicStream := ConvertOpenRouterStreamToAnthropicStream(stream, ExtractOpenRouterProvider(&orProvider))
	for range anthropicStream {
	}
	if orProvider != providerName {
		t.Fatalf("expected %s, got %s", providerName, orProvider)
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_EncryptedReasoning_CustomDelimiter(t *testing.T) {
	prev := viper.GetString("mapping.reasoning.delimiter")
	viper.Set("mapping.reasoning.delimiter", "|")
	defer viper.Set("mapping.reasoning.delimiter", prev)
	chunks := []*openrouter.ChatCompletionChunk{
		{
			ID:    "chatcmpl-1",
			Model: "m",
			Choices: []*openrouter.ChatCompletionChunkChoice{
				{Delta: &openrouter.ChatCompletionChunkChoiceDelta{ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{{Type: openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted, ID: "abc", Data: "def"}}}},
			},
		},
	}
	stream := createMockStream(chunks, nil)
	var sigs []string
	for event, err := range ConvertOpenRouterStreamToAnthropicStream(stream) {
		if err != nil {
			t.Fatalf("%v", err)
		}
		if d, ok := event.(*anthropic.EventContentBlockDelta); ok && d.Delta.Type == anthropic.MessageContentDeltaTypeSignatureDelta {
			sigs = append(sigs, d.Delta.Signature)
		}
	}
	if len(sigs) != 1 || sigs[0] != "abc|def" {
		t.Fatalf("unexpected signatures: %v", sigs)
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_EmptyReasoningAndToolArgs_NoDelta(t *testing.T) {
	chunks := []*openrouter.ChatCompletionChunk{
		{
			ID:    "chatcmpl-1",
			Model: "m",
			Choices: []*openrouter.ChatCompletionChunkChoice{
				{Delta: &openrouter.ChatCompletionChunkChoiceDelta{ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: ""}}, ToolCalls: []*openrouter.ChatCompletionToolCall{{ID: "tool_1", Function: &openrouter.ChatCompletionMessageToolCallFunction{Name: "fn", Arguments: ""}}}}},
			},
		},
	}
	stream := createMockStream(chunks, nil)
	for event, err := range ConvertOpenRouterStreamToAnthropicStream(stream) {
		if err != nil {
			t.Fatalf("%v", err)
		}
		if d, ok := event.(*anthropic.EventContentBlockDelta); ok {
			if d.Delta.Type == anthropic.MessageContentDeltaTypeThinkingDelta || d.Delta.Type == anthropic.MessageContentDeltaTypeInputJSONDelta {
				t.Fatalf("unexpected delta: %v", d.Delta.Type)
			}
		}
	}
}

func TestConvertOpenRouterStreamToAnthropicStream_CrossTypeBlockStop(t *testing.T) {
	chunks := []*openrouter.ChatCompletionChunk{
		{ID: "chatcmpl-1", Model: "m", Choices: []*openrouter.ChatCompletionChunkChoice{{Delta: &openrouter.ChatCompletionChunkChoiceDelta{ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "a"}}}}}},
		{ID: "chatcmpl-1", Model: "m", Choices: []*openrouter.ChatCompletionChunkChoice{{Delta: &openrouter.ChatCompletionChunkChoiceDelta{Content: "b"}}}},
		{ID: "chatcmpl-1", Model: "m", Choices: []*openrouter.ChatCompletionChunkChoice{{Delta: &openrouter.ChatCompletionChunkChoiceDelta{ToolCalls: []*openrouter.ChatCompletionToolCall{{ID: "tool_1", Function: &openrouter.ChatCompletionMessageToolCallFunction{Name: "fn", Arguments: "{}"}}}}}}},
	}
	stream := createMockStream(chunks, nil)
	var starts []*anthropic.EventContentBlockStart
	var stops []*anthropic.EventContentBlockStop
	for event, err := range ConvertOpenRouterStreamToAnthropicStream(stream) {
		if err != nil {
			t.Fatalf("%v", err)
		}
		switch e := event.(type) {
		case *anthropic.EventContentBlockStart:
			starts = append(starts, e)
		case *anthropic.EventContentBlockStop:
			stops = append(stops, e)
		}
	}
	if len(starts) != 3 || len(stops) != 3 {
		t.Fatalf("unexpected counts: %d %d", len(starts), len(stops))
	}
	if starts[0].ContentBlock.Type != anthropic.MessageContentTypeThinking || starts[0].Index != 0 {
		t.Fatalf("unexpected first start")
	}
	if starts[1].ContentBlock.Type != anthropic.MessageContentTypeText || starts[1].Index != 1 {
		t.Fatalf("unexpected second start")
	}
	if starts[2].ContentBlock.Type != anthropic.MessageContentTypeToolUse || starts[2].Index != 2 {
		t.Fatalf("unexpected third start")
	}
	if stops[0].Index != 0 || stops[1].Index != 1 || stops[2].Index != 2 {
		t.Fatalf("unexpected stop indices: %v %v %v", stops[0].Index, stops[1].Index, stops[2].Index)
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
