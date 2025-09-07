package provider

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
)

func TestNewProvider(t *testing.T) {
	provider := NewProvider()
	if provider == nil {
		t.Fatal("NewProvider() returned nil")
	}

	// Verify that ResponseHandler can be created
	handler := provider.responseHandler()
	if handler == nil {
		t.Fatal("ResponseHandler() returned nil")
	}
}

func TestCreateOpenRouterChatCompletion_ClaudeThinking(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("openrouter.api_key") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	// Create test request for Claude 3.7 Sonnet Thinking model
	req := &openrouter.CreateChatCompletionRequest{
		Model: "anthropic/claude-3.7-sonnet:thinking",
		Messages: []*openrouter.ChatCompletionMessage{
			{
				Role: openrouter.ChatCompletionMessageRoleUser,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Hello, how are you?",
				},
			},
		},
		MaxCompletionTokens: lo.ToPtr(100),
		Temperature:         lo.ToPtr(0.7),
		StreamOptions:       &openrouter.ChatCompletionStreamOptions{IncludeUsage: true},
	}

	// Call the API
	stream, header, err := provider.CreateOpenRouterChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("CreateOpenRouterChatCompletion failed: %v", err)
	}

	// Verify header is not nil
	if header == nil {
		t.Fatal("HTTP header is nil")
	}

	// Process the stream and verify data format
	var chunks []*openrouter.ChatCompletionChunk
	var hasContent bool

	for chunk, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		if chunk == nil {
			t.Fatal("Received nil chunk")
		}

		// Verify chunk structure matches expected format
		validateChatCompletionChunk(t, chunk)
		chunks = append(chunks, chunk)

		// Check if we received any content
		if hasChunkContent(chunk) {
			hasContent = true
		}
	}

	// Verify we received at least one chunk
	if len(chunks) == 0 {
		t.Fatal("No chunks received from stream")
	}

	// Verify we received actual content
	if !hasContent {
		t.Error("No content received in any chunk")
	}

	t.Logf("Successfully received %d chunks from Claude Thinking model", len(chunks))
}

func TestCreateOpenRouterChatCompletion_DataFormat(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("openrouter.api_key") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &openrouter.CreateChatCompletionRequest{
		Model: "anthropic/claude-3.7-sonnet:thinking",
		Messages: []*openrouter.ChatCompletionMessage{
			{
				Role: openrouter.ChatCompletionMessageRoleUser,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "What is 2+2?",
				},
			},
		},
		MaxCompletionTokens: lo.ToPtr(50),
		StreamOptions:       &openrouter.ChatCompletionStreamOptions{IncludeUsage: true},
	}

	stream, header, err := provider.CreateOpenRouterChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("API call failed: %v", err)
	}

	if header == nil {
		t.Fatal("Header should not be nil")
	}

	// Use ChatCompletionBuilder to validate the complete response structure
	builder := openrouter.NewChatCompletionBuilder()
	var finalUsage *openrouter.ChatCompletionUsage

	for chunk, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		// Validate chunk format
		validateChatCompletionChunk(t, chunk)

		// Add to builder
		builder.Add(chunk)

		// Track usage information
		if chunk.Usage != nil {
			finalUsage = chunk.Usage
		}
	}

	// Build final completion and validate
	completion := builder.Build()
	validateChatCompletion(t, completion)

	// Verify usage information is present
	if finalUsage == nil {
		t.Error("No usage information received")
	} else {
		if finalUsage.TotalTokens <= 0 {
			t.Error("Total tokens should be positive")
		}
	}

	t.Logf("Completion ID: %s, Model: %s, Provider: %s", completion.ID, completion.Model, completion.Provider)
}

func TestCreateOpenRouterChatCompletion_WithProviderPreference(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("openrouter.api_key") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	// Create provider preference to only use Anthropic models
	providerPref := &openrouter.ProviderPreference{
		Only: []string{openrouter.ProviderAnthropic},
	}

	req := &openrouter.CreateChatCompletionRequest{
		Model: "anthropic/claude-3.7-sonnet:thinking",
		Messages: []*openrouter.ChatCompletionMessage{
			{
				Role: openrouter.ChatCompletionMessageRoleUser,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Hello, please respond briefly.",
				},
			},
		},
		MaxCompletionTokens: lo.ToPtr(100),
		StreamOptions:       &openrouter.ChatCompletionStreamOptions{IncludeUsage: true},
	}

	// Call the API with provider preference
	stream, header, err := provider.CreateOpenRouterChatCompletion(
		ctx,
		req,
		openrouter.WithProviderPreference(providerPref),
	)
	if err != nil {
		t.Fatalf("API call with provider preference failed: %v", err)
	}

	if header == nil {
		t.Fatal("Header should not be nil")
	}

	var chunks []*openrouter.ChatCompletionChunk
	var hasContent bool
	var providerFound bool

	for chunk, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		validateChatCompletionChunk(t, chunk)
		chunks = append(chunks, chunk)

		// Check if we received content
		if hasChunkContent(chunk) {
			hasContent = true
		}

		// Verify that the provider is indeed from Anthropic
		if chunk.Provider != "" {
			providerFound = true
			if !strings.EqualFold(chunk.Provider, string(openrouter.ProviderAnthropic)) {
				t.Errorf("Expected provider to be %q, but got %q", openrouter.ProviderAnthropic, chunk.Provider)
			}
		}
	}

	// Verify we received chunks
	if len(chunks) == 0 {
		t.Fatal("No chunks received from stream")
	}

	// Verify we received content
	if !hasContent {
		t.Error("No content received in any chunk")
	}

	// Verify provider preference was honored
	if !providerFound {
		t.Error("No provider information found in chunks")
	}

	t.Logf("Provider preference test completed successfully:")
	t.Logf("  - Total chunks: %d", len(chunks))
	t.Logf("  - Has content: %v", hasContent)
	t.Logf("  - Provider verified: %v", providerFound)
}

func TestGenerateAnthropicMessage_Basic(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.GenerateMessageRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Hello! Please respond briefly.",
					},
				},
			},
		},
		MaxTokens: 100,
	}

	// Call the Anthropic API
	stream, header, err := provider.GenerateAnthropicMessage(ctx, req)
	if err != nil {
		t.Fatalf("GenerateAnthropicMessage failed: %v", err)
	}

	// Verify header is not nil
	if header == nil {
		t.Fatal("HTTP header is nil")
	}

	// Process the stream and verify data format
	var events []anthropic.Event
	var hasContent bool
	var hasMessageStart bool
	var hasMessageStop bool

	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		if event == nil {
			t.Fatal("Received nil event")
		}

		// Validate event structure
		validateAnthropicEvent(t, event)
		events = append(events, event)

		// Check event types
		switch event.EventType() {
		case anthropic.EventTypeMessageStart:
			hasMessageStart = true
		case anthropic.EventTypeMessageStop:
			hasMessageStop = true
		case anthropic.EventTypeContentBlockDelta:
			hasContent = true
		}
	}

	// Verify we received events
	if len(events) == 0 {
		t.Fatal("No events received from stream")
	}

	// Verify we received expected event types
	if !hasMessageStart {
		t.Error("Did not receive message_start event")
	}

	if !hasMessageStop {
		t.Error("Did not receive message_stop event")
	}

	if !hasContent {
		t.Error("Did not receive content events")
	}

	t.Logf("Successfully received %d events from Anthropic API", len(events))
}

func TestGenerateAnthropicMessage_Thinking(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	// Use a question that should trigger thinking
	req := &anthropic.GenerateMessageRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Think step by step: If I have 15 apples and I give away 7, then buy 3 more, how many apples do I have?",
					},
				},
			},
		},
		MaxTokens: 2048,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 1024,
		},
	}

	stream, header, err := provider.GenerateAnthropicMessage(ctx, req)
	if err != nil {
		t.Fatalf("GenerateAnthropicMessage with thinking failed: %v", err)
	}

	if header == nil {
		t.Fatal("Header should not be nil")
	}

	var events []anthropic.Event
	var hasThinking bool
	var hasTextContent bool
	var thinkingContent []string
	var textContent []string

	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		validateAnthropicEvent(t, event)
		events = append(events, event)

		// Check for thinking and text content
		if blockDelta, ok := event.(*anthropic.EventContentBlockDelta); ok {
			if blockDelta.Delta != nil {
				switch blockDelta.Delta.Type {
				case anthropic.MessageContentDeltaTypeThinkingDelta:
					hasThinking = true
					thinkingContent = append(thinkingContent, blockDelta.Delta.Thinking)
					t.Logf("Received thinking delta: %q", blockDelta.Delta.Thinking)
				case anthropic.MessageContentDeltaTypeTextDelta:
					hasTextContent = true
					textContent = append(textContent, blockDelta.Delta.Text)
				}
			}
		}
	}

	// Verify we received events
	if len(events) == 0 {
		t.Fatal("No events received from stream")
	}

	// Verify we received thinking content
	if !hasThinking {
		t.Error("Expected thinking content from thinking-enabled request, but none was received")
	}

	// Verify we also received text content
	if !hasTextContent {
		t.Error("Should also receive text content alongside thinking")
	}

	t.Logf("Thinking validation results:")
	t.Logf("  - Has thinking: %v (%d deltas)", hasThinking, len(thinkingContent))
	t.Logf("  - Has text content: %v (%d deltas)", hasTextContent, len(textContent))

	if hasThinking {
		totalThinkingLength := 0
		for _, content := range thinkingContent {
			totalThinkingLength += len(content)
		}
		t.Logf("  - Total thinking length: %d characters", totalThinkingLength)

		if totalThinkingLength > 0 {
			t.Logf("  - First thinking delta preview: %q",
				truncateString(thinkingContent[0], 100))
		}
	}
}

func TestCreateOpenRouterChatCompletion_ReasoningValidation(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("openrouter.api_key") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	// Use a question that should trigger reasoning
	req := &openrouter.CreateChatCompletionRequest{
		Model: "anthropic/claude-3.7-sonnet:thinking",
		Messages: []*openrouter.ChatCompletionMessage{
			{
				Role: openrouter.ChatCompletionMessageRoleUser,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Solve this step by step: If a train leaves station A at 2 PM traveling at 60 mph, and another train leaves station B (120 miles away) at 2:30 PM traveling toward station A at 80 mph, at what time will they meet?",
				},
			},
		},
		Reasoning:           &openrouter.ChatCompletionReasoning{MaxTokens: 1024, Enabled: true},
		MaxCompletionTokens: lo.ToPtr(2048),
		StreamOptions:       &openrouter.ChatCompletionStreamOptions{IncludeUsage: true},
	}

	stream, header, err := provider.CreateOpenRouterChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("API call failed: %v", err)
	}

	if header == nil {
		t.Fatal("Header should not be nil")
	}

	var hasReasoning bool
	var hasContent bool
	var reasoningChunks []string
	var contentChunks []string

	for chunk, streamErr := range stream {
		if streamErr != nil {
			t.Fatalf("Stream error: %v", streamErr)
		}

		validateChatCompletionChunk(t, chunk)

		// Check for reasoning and content in each choice
		for _, choice := range chunk.Choices {
			if delta := choice.Delta; delta != nil {
				// Collect reasoning content
				if delta.Reasoning != "" {
					hasReasoning = true
					reasoningChunks = append(reasoningChunks, delta.Reasoning)
					t.Logf("Received reasoning chunk: %q", delta.Reasoning)
				}

				// Collect regular content
				if delta.Content != "" {
					hasContent = true
					contentChunks = append(contentChunks, delta.Content)
				}
			}
		}
	}

	// Verify that we received reasoning content
	if !hasReasoning {
		t.Error("Claude 3.7 Sonnet Thinking model should produce reasoning content, but none was received")
	}

	// Verify that we also received regular content
	if !hasContent {
		t.Error("Should also receive regular content alongside reasoning")
	}

	t.Logf("Reasoning validation results:")
	t.Logf("  - Has reasoning: %v (%d chunks)", hasReasoning, len(reasoningChunks))
	t.Logf("  - Has content: %v (%d chunks)", hasContent, len(contentChunks))

	if hasReasoning {
		totalReasoningLength := 0
		for _, chunk := range reasoningChunks {
			totalReasoningLength += len(chunk)
		}
		t.Logf("  - Total reasoning length: %d characters", totalReasoningLength)

		if totalReasoningLength > 0 {
			t.Logf("  - First reasoning chunk preview: %q",
				truncateString(reasoningChunks[0], 100))
		}
	}
}

// validateChatCompletionChunk validates the structure of a ChatCompletionChunk
func TestCreateOpenRouterChatCompletion_CachedTokensAcrossRequests(t *testing.T) {
	if viper.GetString("openrouter.api_key") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping integration test")
	}
	provider := NewProvider()
	ctx := context.Background()
	mkReq := func() *openrouter.CreateChatCompletionRequest {
		lp := strings.Repeat("这是一段用于缓存测试的长提示。", 500)
		return &openrouter.CreateChatCompletionRequest{
			Model: "anthropic/claude-sonnet-4",
			Messages: []*openrouter.ChatCompletionMessage{
				{
					Role: openrouter.ChatCompletionMessageRoleUser,
					Content: &openrouter.ChatCompletionMessageContent{
						Type: openrouter.ChatCompletionMessageContentTypeParts,
						Parts: []*openrouter.ChatCompletionMessageContentPart{
							{
								Type: openrouter.ChatCompletionMessageContentPartTypeText,
								Text: lp,
								CacheControl: &openrouter.ChatCompletionMessageCacheControl{Type: "ephemeral", TTL: "5m"},
							},
						},
					},
				},
			},
			MaxCompletionTokens: lo.ToPtr(64),
			StreamOptions:       &openrouter.ChatCompletionStreamOptions{IncludeUsage: true},
		}
	}
	getCached := func() (int64, error) {
		pref := openrouter.ProviderPreference{Only: []string{openrouter.ProviderAnthropic}}
		stream, _, err := provider.CreateOpenRouterChatCompletion(ctx, mkReq(), openrouter.WithProviderPreference(&pref))
		if err != nil {
			return 0, err
		}
		var cached int64
		for chunk, streamErr := range stream {
			if streamErr != nil {
				return 0, streamErr
			}
			if chunk != nil && chunk.Usage != nil && chunk.Usage.PromptTokensDetails != nil {
				cached = chunk.Usage.PromptTokensDetails.CachedTokens
			}
		}
		return cached, nil
	}
	first, err := getCached()
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	second, err := getCached()
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	t.Logf("cached_tokens first=%d second=%d", first, second)
	if second == 0 && first == 0 {
		t.Skipf("provider did not report cached_tokens; first=%d second=%d", first, second)
	}
	if second < first {
		t.Fatalf("expected cached tokens to be non-decreasing, got first=%d second=%d", first, second)
	}
}

func validateChatCompletionChunk(t *testing.T, chunk *openrouter.ChatCompletionChunk) {
	t.Helper()

	// Basic required fields
	if chunk.ID == "" {
		t.Error("Chunk ID should not be empty")
	}

	if chunk.Model == "" {
		t.Error("Chunk Model should not be empty")
	}

	if chunk.Object == "" {
		t.Error("Chunk Object should not be empty")
	}

	if chunk.Created <= 0 {
		t.Error("Chunk Created timestamp should be positive")
	}

	// Verify Provider field is present (new field)
	if chunk.Provider == "" {
		t.Error("Chunk Provider should not be empty")
	}

	// Validate choices structure
	if len(chunk.Choices) == 0 {
		t.Error("Chunk should have at least one choice")
	}

	for i, choice := range chunk.Choices {
		if choice.Index != i {
			t.Errorf("Choice index mismatch: expected %d, got %d", i, choice.Index)
		}

		// Validate delta structure if present
		if choice.Delta != nil {
			validateChunkChoiceDelta(t, choice.Delta)
		}
	}
}

// validateChunkChoiceDelta validates the structure of a ChatCompletionChunkChoiceDelta
func validateChunkChoiceDelta(t *testing.T, delta *openrouter.ChatCompletionChunkChoiceDelta) {
	t.Helper()

	if !hasDeltaContent(delta) {
		// This is okay for some chunks, so just log
		t.Logf("Delta has no content (this may be normal for some chunks)")
	}
}

// validateChatCompletion validates the structure of a complete ChatCompletion
func validateChatCompletion(t *testing.T, completion *openrouter.ChatCompletion) {
	t.Helper()

	if completion.ID == "" {
		t.Error("Completion ID should not be empty")
	}

	if completion.Model == "" {
		t.Error("Completion Model should not be empty")
	}

	if completion.Provider == "" {
		t.Error("Completion Provider should not be empty")
	}

	if completion.Created <= 0 {
		t.Error("Completion Created timestamp should be positive")
	}

	if len(completion.Choices) == 0 {
		t.Error("Completion should have at least one choice")
	}

	for _, choice := range completion.Choices {
		if choice.Message == nil {
			t.Error("Choice message should not be nil")
		} else {
			if choice.Message.Content == nil {
				t.Error("Message content should not be nil")
			}
			if choice.Message.Role == "" {
				t.Error("Message role should not be empty")
			}
		}
	}
}

// hasDeltaContent checks if a delta contains any meaningful content
func hasDeltaContent(delta *openrouter.ChatCompletionChunkChoiceDelta) bool {
	if delta.Content != "" {
		return true
	}
	if delta.Refusal != nil && *delta.Refusal != "" {
		return true
	}
	if delta.Role != "" {
		return true
	}
	if len(delta.ToolCalls) > 0 {
		for _, toolCall := range delta.ToolCalls {
			if toolCall.Function != nil && toolCall.Function.Arguments != "" {
				return true
			}
		}
	}
	if delta.Reasoning != "" {
		return true
	}
	if len(delta.ReasoningDetails) > 0 {
		return true
	}
	return false
}

// hasChunkContent checks if a chunk contains any meaningful content
func hasChunkContent(chunk *openrouter.ChatCompletionChunk) bool {
	for _, choice := range chunk.Choices {
		if delta := choice.Delta; delta != nil {
			if hasDeltaContent(delta) {
				return true
			}
		}
	}
	return false
}

// validateAnthropicEvent validates the structure of an Anthropic Event
func validateAnthropicEvent(t *testing.T, event anthropic.Event) {
	t.Helper()

	if event == nil {
		t.Error("Event should not be nil")
		return
	}

	// Check that event type is valid
	eventType := event.EventType()
	switch eventType {
	case anthropic.EventTypePing,
		anthropic.EventTypeError,
		anthropic.EventTypeMessageStart,
		anthropic.EventTypeMessageDelta,
		anthropic.EventTypeMessageStop,
		anthropic.EventTypeContentBlockStart,
		anthropic.EventTypeContentBlockDelta,
		anthropic.EventTypeContentBlockStop:
		// Valid event type
	default:
		t.Errorf("Unknown event type: %s", eventType)
	}

	// Validate specific event types
	switch e := event.(type) {
	case *anthropic.EventMessageStart:
		if e.Message == nil {
			t.Error("MessageStart event should have a message")
		}
	case *anthropic.EventContentBlockDelta:
		if e.Delta == nil {
			t.Error("ContentBlockDelta event should have a delta")
		}
	case *anthropic.EventError:
		if e.Error == nil {
			t.Error("Error event should have an error")
		}
	}
}

func TestCountAnthropicTokens_Basic(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.CountTokensRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Hello, how many tokens is this message?",
					},
				},
			},
		},
	}

	// Call the CountAnthropicTokens API
	usage, err := provider.CountAnthropicTokens(ctx, req)
	if err != nil {
		t.Fatalf("CountAnthropicTokens failed: %v", err)
	}

	// Verify usage is not nil
	if usage == nil {
		t.Fatal("Usage should not be nil")
	}

	// Verify token counts are positive
	if usage.InputTokens <= 0 {
		t.Error("Input tokens should be positive")
	}

	if usage.OutputTokens < 0 {
		t.Error("Output tokens should be non-negative")
	}

	// For count tokens endpoint, output tokens should typically be 0
	if usage.OutputTokens != 0 {
		t.Logf("Note: Output tokens is %d, expected 0 for count endpoint", usage.OutputTokens)
	}

	t.Logf("Token count results:")
	t.Logf("  - Input tokens: %d", usage.InputTokens)
	t.Logf("  - Output tokens: %d", usage.OutputTokens)
}

func TestCountAnthropicTokens_WithSystem(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.CountTokensRequest{
		Model:  "claude-3-5-sonnet-20241022",
		System: anthropic.MessageContents{{Type: anthropic.MessageContentTypeText, Text: "You are a helpful assistant that provides concise answers."}},
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "What's the weather like?",
					},
				},
			},
		},
	}

	usage, err := provider.CountAnthropicTokens(ctx, req)
	if err != nil {
		t.Fatalf("CountAnthropicTokens with system message failed: %v", err)
	}

	if usage == nil {
		t.Fatal("Usage should not be nil")
	}

	// Should have more tokens due to system message
	if usage.InputTokens <= 0 {
		t.Error("Input tokens should be positive")
	}

	t.Logf("Token count with system message:")
	t.Logf("  - Input tokens: %d", usage.InputTokens)
	t.Logf("  - Output tokens: %d", usage.OutputTokens)
}

func TestCountAnthropicTokens_WithThinking(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.CountTokensRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Think step by step: What is 15 + 27?",
					},
				},
			},
		},
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 1024,
		},
	}

	usage, err := provider.CountAnthropicTokens(ctx, req)
	if err != nil {
		t.Fatalf("CountAnthropicTokens with thinking failed: %v", err)
	}

	if usage == nil {
		t.Fatal("Usage should not be nil")
	}

	if usage.InputTokens <= 0 {
		t.Error("Input tokens should be positive")
	}

	t.Logf("Token count with thinking:")
	t.Logf("  - Input tokens: %d", usage.InputTokens)
	t.Logf("  - Output tokens: %d", usage.OutputTokens)
}

func TestCountAnthropicTokens_InvalidModel(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.CountTokensRequest{
		Model: "invalid-model-name",
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: []*anthropic.MessageContent{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Hello",
					},
				},
			},
		},
	}

	usage, err := provider.CountAnthropicTokens(ctx, req)
	if err == nil {
		t.Fatal("Expected error for invalid model, but got none")
	}

	// Note: Due to the generated code structure, usage might not be nil even on error
	if usage != nil {
		t.Logf("Usage is not nil on error (this is expected with the current generated code)")
	}

	// Verify it contains an Anthropic error
	var anthropicErr *anthropic.Error
	if !errors.As(err, &anthropicErr) {
		t.Errorf("Expected to be able to unwrap Anthropic error, got: %T, message: %v", err, err)
	} else {
		t.Logf("Successfully unwrapped Anthropic error: %v", anthropicErr)
		if anthropicErr.Type() != "not_found_error" {
			t.Errorf("Expected not_found_error, got: %s", anthropicErr.Type())
		}
	}
}

func TestCountAnthropicTokens_EmptyMessages(t *testing.T) {
	// Skip test if no API key is provided
	if viper.GetString("anthropic.api_key") == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	ctx := context.Background()

	req := &anthropic.CountTokensRequest{
		Model:    "claude-3-5-sonnet-20241022",
		Messages: []*anthropic.Message{}, // Empty messages
	}

	usage, err := provider.CountAnthropicTokens(ctx, req)
	if err == nil {
		t.Fatal("Expected error for empty messages, but got none")
	}

	// Note: Due to the generated code structure, usage might not be nil even on error
	if usage != nil {
		t.Logf("Usage is not nil on error (this is expected with the current generated code)")
	}

	// Verify it contains an Anthropic error
	var anthropicErr *anthropic.Error
	if !errors.As(err, &anthropicErr) {
		t.Errorf("Expected to be able to unwrap Anthropic error, got: %T, message: %v", err, err)
	} else {
		t.Logf("Successfully unwrapped Anthropic error: %v", anthropicErr)
		// Empty messages should result in validation error
		if anthropicErr.Type() == "" {
			t.Errorf("Expected non-empty error type")
		}
	}
}

// truncateString truncates a string to the specified length and adds "..." if truncated
func truncateString(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}
	return s[:maxLength] + "..."
}
