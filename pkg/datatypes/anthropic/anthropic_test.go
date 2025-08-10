package anthropic

import (
	"bytes"
	"testing"

	"github.com/samber/lo"
)

// createTestStream creates a test stream with the given events
func createTestStream(events []Event, errors []error) MessageStream {
	return func(yield func(Event, error) bool) {
		for i, event := range events {
			var err error
			if i < len(errors) {
				err = errors[i]
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

func TestMessageBuilder_BasicTextMessage(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_123",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 10},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeText,
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type: MessageContentDeltaTypeTextDelta,
				Text: "Hello ",
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type: MessageContentDeltaTypeTextDelta,
				Text: "world!",
			},
		},
		&EventContentBlockStop{Index: 0},
		&EventMessageDelta{
			Delta: &Message{
				StopReason: lo.ToPtr(StopReasonEndTurn),
			},
			Usage: &Usage{OutputTokens: 5},
		},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if result.ID != "msg_123" {
		t.Errorf("Expected ID 'msg_123', got: %s", result.ID)
	}
	if result.Model != "claude-3-5-sonnet-20241022" {
		t.Errorf("Expected Model 'claude-3-5-sonnet-20241022', got: %s", result.Model)
	}
	if result.Type != MessageTypeMessage {
		t.Errorf("Expected Type %v, got: %v", MessageTypeMessage, result.Type)
	}
	if result.Role != MessageRoleAssistant {
		t.Errorf("Expected Role %v, got: %v", MessageRoleAssistant, result.Role)
	}
	if result.StopReason == nil || *result.StopReason != StopReasonEndTurn {
		t.Errorf("Expected StopReason %v, got: %v", StopReasonEndTurn, result.StopReason)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("Expected InputTokens 10, got: %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 5 {
		t.Errorf("Expected OutputTokens 5, got: %d", result.Usage.OutputTokens)
	}
	if len(result.Content) != 1 {
		t.Fatalf("Expected 1 content block, got: %d", len(result.Content))
	}
	if result.Content[0].Type != MessageContentTypeText {
		t.Errorf("Expected content type %v, got: %v", MessageContentTypeText, result.Content[0].Type)
	}
	if result.Content[0].Text != "Hello world!" {
		t.Errorf("Expected text 'Hello world!', got: %s", result.Content[0].Text)
	}
}

func TestMessageBuilder_ThinkingMessage(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_456",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 15},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeThinking,
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:     MessageContentDeltaTypeThinkingDelta,
				Thinking: "Let me think about this... ",
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:     MessageContentDeltaTypeThinkingDelta,
				Thinking: "I need to consider all aspects.",
			},
		},
		&EventContentBlockStop{Index: 0},
		&EventContentBlockStart{
			Index: 1,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeText,
			},
		},
		&EventContentBlockDelta{
			Index: 1,
			Delta: &MessageContentDelta{
				Type: MessageContentDeltaTypeTextDelta,
				Text: "Based on my analysis...",
			},
		},
		&EventContentBlockStop{Index: 1},
		&EventMessageDelta{
			Delta: &Message{
				StopReason: lo.ToPtr(StopReasonEndTurn),
			},
			Usage: &Usage{OutputTokens: 25},
		},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if len(result.Content) != 2 {
		t.Fatalf("Expected 2 content blocks, got: %d", len(result.Content))
	}

	// First content block should be thinking
	if result.Content[0].Type != MessageContentTypeThinking {
		t.Errorf("Expected first content type %v, got: %v", MessageContentTypeThinking, result.Content[0].Type)
	}
	expectedThinking := "Let me think about this... I need to consider all aspects."
	if result.Content[0].Thinking != expectedThinking {
		t.Errorf("Expected thinking '%s', got: '%s'", expectedThinking, result.Content[0].Thinking)
	}

	// Second content block should be text
	if result.Content[1].Type != MessageContentTypeText {
		t.Errorf("Expected second content type %v, got: %v", MessageContentTypeText, result.Content[1].Type)
	}
	if result.Content[1].Text != "Based on my analysis..." {
		t.Errorf("Expected text 'Based on my analysis...', got: '%s'", result.Content[1].Text)
	}
}

func TestMessageBuilder_ToolUseMessage(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_789",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 20},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeToolUse,
				ID:   "tool_123",
				Name: "get_weather",
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:        MessageContentDeltaTypeInputJSONDelta,
				PartialJSON: `{"city": `,
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:        MessageContentDeltaTypeInputJSONDelta,
				PartialJSON: `"San Francisco"}`,
			},
		},
		&EventContentBlockStop{Index: 0},
		&EventMessageDelta{
			Delta: &Message{
				StopReason: lo.ToPtr(StopReasonToolUse),
			},
			Usage: &Usage{OutputTokens: 15},
		},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if len(result.Content) != 1 {
		t.Fatalf("Expected 1 content block, got: %d", len(result.Content))
	}

	content := result.Content[0]
	if content.Type != MessageContentTypeToolUse {
		t.Errorf("Expected content type %v, got: %v", MessageContentTypeToolUse, content.Type)
	}
	if content.ID != "tool_123" {
		t.Errorf("Expected ID 'tool_123', got: %s", content.ID)
	}
	if content.Name != "get_weather" {
		t.Errorf("Expected Name 'get_weather', got: %s", content.Name)
	}
	expectedInput := []byte(`{"city":"San Francisco"}`)
	if !bytes.Equal(content.Input, expectedInput) {
		t.Errorf("Expected Input %s, got: %s", expectedInput, content.Input)
	}
	if result.StopReason == nil || *result.StopReason != StopReasonToolUse {
		t.Errorf("Expected StopReason %v, got: %v", StopReasonToolUse, result.StopReason)
	}
}

func TestMessageBuilder_StreamError(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_error",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 5},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeText,
			},
		},
	}

	// Stream errors should be handled separately from the builder
	// The builder only processes events, not stream transport errors
	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	// Stream transport errors are handled at a higher level, not in the builder
}

func TestMessageBuilder_EventError(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_event_error",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 5},
			},
		},
		&EventError{
			Error: &StreamError{
				ErrType:    "rate_limit_error",
				ErrMessage: "Rate limit exceeded",
			},
		},
	}

	builder := NewMessageBuilder()
	var err error
	for _, event := range events {
		err = builder.Add(event)
		if err != nil {
			break // EventError should cause Add to return an error
		}
	}

	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	// Should be wrapped in Anthropic Error type
	var anthropicErr *Error
	anthropicErr, ok := err.(*Error)
	if !ok {
		t.Fatalf("Expected *Error type, got: %T", err)
	}
	if anthropicErr.ContentType != ErrorContentType {
		t.Errorf("Expected ContentType %v, got: %v", ErrorContentType, anthropicErr.ContentType)
	}
	if anthropicErr.Inner.Type != "rate_limit_error" {
		t.Errorf("Expected Type 'rate_limit_error', got: %s", anthropicErr.Inner.Type)
	}
	if anthropicErr.Inner.Message != "Rate limit exceeded" {
		t.Errorf("Expected Message 'Rate limit exceeded', got: %s", anthropicErr.Inner.Message)
	}
}

func TestMessageBuilder_EmptyStream(t *testing.T) {
	builder := NewMessageBuilder()
	// No events added
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if result.Type != MessageTypeMessage {
		t.Errorf("Expected Type %v, got: %v", MessageTypeMessage, result.Type)
	}
	if result.Role != MessageRoleAssistant {
		t.Errorf("Expected Role %v, got: %v", MessageRoleAssistant, result.Role)
	}
	if result.ID != "" {
		t.Errorf("Expected empty ID, got: %s", result.ID)
	}
	if result.Model != "" {
		t.Errorf("Expected empty Model, got: %s", result.Model)
	}
	if len(result.Content) != 0 {
		t.Errorf("Expected empty Content, got: %v", result.Content)
	}
	if result.Usage == nil {
		t.Fatal("Expected non-nil Usage")
	}
	if result.Usage.InputTokens != 0 {
		t.Errorf("Expected InputTokens 0, got: %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 0 {
		t.Errorf("Expected OutputTokens 0, got: %d", result.Usage.OutputTokens)
	}
}

func TestMessageBuilder_MultipleContentBlocks(t *testing.T) {
	// Test multiple content blocks with proper indexes
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_multi",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 30},
			},
		},
		// First text block at index 0
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeText,
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type: MessageContentDeltaTypeTextDelta,
				Text: "First block content",
			},
		},
		&EventContentBlockStop{Index: 0},
		// Second text block at index 1
		&EventContentBlockStart{
			Index: 1,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeText,
			},
		},
		&EventContentBlockDelta{
			Index: 1,
			Delta: &MessageContentDelta{
				Type: MessageContentDeltaTypeTextDelta,
				Text: "Second block content",
			},
		},
		&EventContentBlockStop{Index: 1},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if len(result.Content) != 2 {
		t.Fatalf("Expected 2 content blocks, got: %d", len(result.Content))
	}

	// First block
	if result.Content[0].Type != MessageContentTypeText {
		t.Errorf("Expected text content type at index 0, got: %v", result.Content[0].Type)
	}
	if result.Content[0].Text != "First block content" {
		t.Errorf("Expected text 'First block content', got: %s", result.Content[0].Text)
	}

	// Second block
	if result.Content[1].Type != MessageContentTypeText {
		t.Errorf("Expected text content type at index 1, got: %v", result.Content[1].Type)
	}
	if result.Content[1].Text != "Second block content" {
		t.Errorf("Expected text 'Second block content', got: %s", result.Content[1].Text)
	}
}

func TestMessageBuilder_NilEventFields(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: nil, // Nil message
		},
		&EventContentBlockStart{
			Index:        0,
			ContentBlock: nil, // Nil content block
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: nil, // Nil delta
		},
		&EventMessageDelta{
			Delta: nil, // Nil delta
			Usage: nil, // Nil usage
		},
		&EventError{
			Error: nil, // Nil error - should not cause error return
		},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if result.Type != MessageTypeMessage {
		t.Errorf("Expected Type %v, got: %v", MessageTypeMessage, result.Type)
	}
	if result.Role != MessageRoleAssistant {
		t.Errorf("Expected Role %v, got: %v", MessageRoleAssistant, result.Role)
	}
}

func TestMessageBuilder_RedactedThinkingPanic(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_redacted",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 5},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeRedactedThinking,
			},
		},
	}

	builder := NewMessageBuilder()

	// Should panic with "unreachable redacted_thinking"
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic, but didn't panic")
		} else if r != "unreachable redacted_thinking" {
			t.Errorf("Expected panic 'unreachable redacted_thinking', got: %v", r)
		}
	}()

	for _, event := range events {
		builder.Add(event) // This should panic on the redacted thinking event
	}
}

func TestMessageBuilder_SignatureDelta(t *testing.T) {
	events := []Event{
		&EventMessageStart{
			Message: &Message{
				ID:    "msg_signature",
				Model: "claude-3-5-sonnet-20241022",
				Usage: &Usage{InputTokens: 10},
			},
		},
		&EventContentBlockStart{
			Index: 0,
			ContentBlock: &MessageContent{
				Type: MessageContentTypeToolUse,
				ID:   "tool_sig",
				Name: "test_tool",
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:      MessageContentDeltaTypeSignatureDelta,
				Signature: "sig123",
			},
		},
		&EventContentBlockDelta{
			Index: 0,
			Delta: &MessageContentDelta{
				Type:        MessageContentDeltaTypeInputJSONDelta,
				PartialJSON: `{"arg": "value"}`,
			},
		},
		&EventContentBlockStop{Index: 0},
	}

	builder := NewMessageBuilder()
	for _, event := range events {
		err := builder.Add(event)
		if err != nil {
			t.Fatalf("Expected no error adding event, got: %v", err)
		}
	}
	result := builder.Message()

	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if len(result.Content) != 1 {
		t.Fatalf("Expected 1 content block, got: %d", len(result.Content))
	}

	content := result.Content[0]
	if content.Type != MessageContentTypeToolUse {
		t.Errorf("Expected content type %v, got: %v", MessageContentTypeToolUse, content.Type)
	}
	if content.Signature != "sig123" {
		t.Errorf("Expected signature 'sig123', got: %s", content.Signature)
	}
}
