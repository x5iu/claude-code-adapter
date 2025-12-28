package adapter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/samber/lo"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openai"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/profile"
)

// testProfile creates a test profile with default settings
func testProfile() *profile.Profile {
	return &profile.Profile{
		Name:     "test",
		Provider: "openrouter",
		Options: &profile.OptionsConfig{
			Strict:                    false,
			ContextWindowResizeFactor: 1.0,
			Reasoning: &profile.ReasoningConfig{
				Format:    "anthropic-claude-v1",
				Effort:    "",
				Delimiter: "/",
			},
		},
		Anthropic: &profile.AnthropicConfig{
			BaseURL: "https://api.anthropic.com",
			Version: "2023-06-01",
		},
		OpenRouter: &profile.OpenRouterConfig{
			BaseURL: "https://openrouter.ai/api",
		},
	}
}

// testCtx creates a context with a test profile
func testCtx() context.Context {
	return profile.WithProfile(context.Background(), testProfile())
}

// testProfileWithOptions creates a test profile with custom options
func testProfileWithOptions(opts func(*profile.Profile)) *profile.Profile {
	p := testProfile()
	opts(p)
	return p
}

// testCtxWithOptions creates a context with a custom test profile
func testCtxWithOptions(opts func(*profile.Profile)) context.Context {
	return profile.WithProfile(context.Background(), testProfileWithOptions(opts))
}

// testCtxWithReasoningFormat creates a test context with specific reasoning format and effort
func testCtxWithReasoningFormat(format, effort string) context.Context {
	return testCtxWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = format
		p.Options.Reasoning.Effort = effort
	})
}

// testCtxWithForceThinking creates a test context with force thinking enabled
func testCtxWithForceThinking(format string) context.Context {
	return testCtxWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = format
		p.Anthropic.ForceThinking = true
	})
}

func TestConvertAnthropicRequestToOpenRouterRequest_BasicFields(t *testing.T) {
	tests := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(*openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "basic fields conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:       "claude-3-5-sonnet-20241022",
				MaxTokens:   1000,
				Temperature: 0.7,
				TopK:        lo.ToPtr(10),
				TopP:        lo.ToPtr(0.9),
				System:      anthropic.MessageContents{{Type: anthropic.MessageContentTypeText, Text: "You are a helpful assistant"}},
				Messages:    []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.Model == "claude-3-5-sonnet-20241022" &&
					*dst.MaxTokens == 1000 &&
					*dst.Temperature == 0.7 &&
					*dst.TopK == 10 &&
					*dst.TopP == 0.9
			},
		},
		{
			name: "MaxTokens dual field mapping",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 2048,
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				// Verify that MaxTokens from Anthropic is mapped to both fields
				return dst.MaxTokens != nil && *dst.MaxTokens == 2048
			},
		},
		{
			name: "zero MaxTokens handling",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 0,
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				// Both fields should be set to 0
				return dst.MaxTokens != nil && *dst.MaxTokens == 0
			},
		},
		{
			name: "with metadata user ID",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Metadata: &anthropic.Metadata{
					UserID: "user123",
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.User == "user123"
			},
		},
		{
			name: "with stop sequences",
			src: &anthropic.GenerateMessageRequest{
				Model:         "claude-3-5-sonnet-20241022",
				MaxTokens:     500,
				StopSequences: []string{"\n", "STOP"},
				Messages:      []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return len(dst.Stop) == 2 &&
					dst.Stop[0] == "\n" &&
					dst.Stop[1] == "STOP"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tt.src)
			if !tt.want(got) {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() validation failed")
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ToolChoice(t *testing.T) {
	tests := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(*openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "tool choice auto",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				ToolChoice: &anthropic.ToolChoice{
					Type:                   anthropic.ToolChoiceTypeAuto,
					DisableParallelToolUse: false,
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return *dst.ParallelToolCalls == true &&
					dst.ToolChoice != nil &&
					dst.ToolChoice.Mode == openrouter.ChatCompletionToolChoiceTypeAuto
			},
		},
		{
			name: "tool choice none",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				ToolChoice: &anthropic.ToolChoice{
					Type:                   anthropic.ToolChoiceTypeNone,
					DisableParallelToolUse: true,
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return *dst.ParallelToolCalls == false &&
					dst.ToolChoice != nil &&
					dst.ToolChoice.Mode == openrouter.ChatCompletionToolChoiceTypeNone
			},
		},
		{
			name: "tool choice any",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				ToolChoice: &anthropic.ToolChoice{
					Type: anthropic.ToolChoiceTypeAny,
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.ToolChoice != nil &&
					dst.ToolChoice.Mode == openrouter.ChatCompletionToolChoiceTypeRequired
			},
		},
		{
			name: "tool choice specific tool",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				ToolChoice: &anthropic.ToolChoice{
					Type: anthropic.ToolChoiceTypeTool,
					Name: "get_weather",
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.ToolChoice != nil &&
					dst.ToolChoice.Tool != nil &&
					dst.ToolChoice.Tool.Type == openrouter.ChatCompletionMessageToolCallTypeFunction &&
					dst.ToolChoice.Tool.Function.Name == "get_weather"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tt.src)
			if !tt.want(got) {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() tool choice validation failed")
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_Tools(t *testing.T) {
	// Set up profile with strict mode enabled
	ctx := testCtxWithOptions(func(p *profile.Profile) {
		p.Options.Strict = true
	})

	inputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The city and state",
			},
		},
		"required": []string{"location"},
	}

	inputSchemaBytes, _ := json.Marshal(inputSchema)

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Tools: []*anthropic.Tool{
			{
				Type:        lo.ToPtr(anthropic.ToolTypeCustom),
				Name:        "get_weather",
				Description: "Get weather for a location",
				InputSchema: json.RawMessage(inputSchemaBytes),
			},
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	if len(got.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(got.Tools))
	}

	tool := got.Tools[0]
	if tool.Type != openrouter.ChatCompletionMessageToolCallTypeFunction {
		t.Errorf("Expected tool type function, got %s", tool.Type)
	}

	if tool.Function.Name != "get_weather" {
		t.Errorf("Expected tool name get_weather, got %s", tool.Function.Name)
	}

	if tool.Function.Description != "Get weather for a location" {
		t.Errorf("Expected tool description 'Get weather for a location', got %s", tool.Function.Description)
	}

	if !tool.Function.Strict {
		t.Error("Expected strict mode to be true")
	}

	// Verify the parameters are properly marshaled
	expectedJSON, _ := json.Marshal(inputSchema)
	if string(tool.Function.Parameters) != string(expectedJSON) {
		t.Errorf("Parameters mismatch. Expected %s, got %s", string(expectedJSON), string(tool.Function.Parameters))
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_Thinking(t *testing.T) {
	tests := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(*openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "thinking enabled",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Thinking: &anthropic.Thinking{
					Type:         anthropic.ThinkingTypeEnabled,
					BudgetTokens: 1000,
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.Reasoning != nil &&
					dst.Reasoning.Enabled == true &&
					dst.Reasoning.MaxTokens == 1000
			},
		},
		{
			name: "thinking disabled",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Thinking: &anthropic.Thinking{
					Type:         anthropic.ThinkingTypeDisabled,
					BudgetTokens: 500,
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.Reasoning != nil &&
					dst.Reasoning.Enabled == false &&
					dst.Reasoning.MaxTokens == 500
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tt.src)
			if !tt.want(got) {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() thinking validation failed")
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_Messages(t *testing.T) {
	tests := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(*openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "system message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				System:    anthropic.MessageContents{{Type: anthropic.MessageContentTypeText, Text: "You are a helpful assistant"}},
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				systemMsg := dst.Messages[0]
				return systemMsg.Role == openrouter.ChatCompletionMessageRoleSystem &&
					systemMsg.Content.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(systemMsg.Content.Parts) == 1 &&
					systemMsg.Content.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					systemMsg.Content.Parts[0].Text == "You are a helpful assistant"
			},
		},
		{
			name: "text message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeText,
								Text: "Hello, how are you?",
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				userMsg := dst.Messages[0] // No system message since System field is empty
				return userMsg.Role == openrouter.ChatCompletionMessageRoleUser &&
					userMsg.Content.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(userMsg.Content.Parts) == 1 &&
					userMsg.Content.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					userMsg.Content.Parts[0].Text == "Hello, how are you?"
			},
		},
		{
			name: "thinking message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleAssistant,
						Content: anthropic.MessageContents{
							{
								Type:      anthropic.MessageContentTypeThinking,
								Thinking:  "Let me think about this...",
								Signature: "sig123",
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				thinkingMsg := dst.Messages[0] // No system message since System field is empty
				return thinkingMsg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					thinkingMsg.Reasoning == "Let me think about this..." &&
					len(thinkingMsg.ReasoningDetails) == 1 &&
					thinkingMsg.ReasoningDetails[0].Type == openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText &&
					thinkingMsg.ReasoningDetails[0].Text == "Let me think about this..." &&
					thinkingMsg.ReasoningDetails[0].Signature == "sig123" &&
					thinkingMsg.ReasoningDetails[0].Format == openrouter.ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1
			},
		},
		{
			name: "multiple thinking message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleAssistant,
						Content: anthropic.MessageContents{
							{
								Type:      anthropic.MessageContentTypeThinking,
								Thinking:  "First, let me analyze the problem...",
								Signature: "sig123",
							},
							{
								Type:      anthropic.MessageContentTypeThinking,
								Thinking:  "Now, let me consider the alternatives...",
								Signature: "sig456",
							},
							{
								Type:      anthropic.MessageContentTypeThinking,
								Thinking:  "Finally, I'll conclude with...",
								Signature: "sig789",
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				thinkingMsg := dst.Messages[0]
				return thinkingMsg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					thinkingMsg.Reasoning == "First, let me analyze the problem..." && // First reasoning becomes the main reasoning
					len(thinkingMsg.ReasoningDetails) == 3 &&
					thinkingMsg.ReasoningDetails[0].Index == 0 &&
					thinkingMsg.ReasoningDetails[0].Text == "First, let me analyze the problem..." &&
					thinkingMsg.ReasoningDetails[0].Signature == "sig123" &&
					thinkingMsg.ReasoningDetails[1].Index == 1 &&
					thinkingMsg.ReasoningDetails[1].Text == "Now, let me consider the alternatives..." &&
					thinkingMsg.ReasoningDetails[1].Signature == "sig456" &&
					thinkingMsg.ReasoningDetails[2].Index == 2 &&
					thinkingMsg.ReasoningDetails[2].Text == "Finally, I'll conclude with..." &&
					thinkingMsg.ReasoningDetails[2].Signature == "sig789"
			},
		},
		{
			name: "image message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeImage,
								Source: &anthropic.MessageContentSource{
									Type:      "base64",
									MediaType: "image/jpeg",
									Data:      "<BASE64_IMAGE_DATA>",
								},
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				imgMsg := dst.Messages[0] // No system message since System field is empty
				return imgMsg.Role == openrouter.ChatCompletionMessageRoleUser &&
					imgMsg.Content.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(imgMsg.Content.Parts) == 1 &&
					imgMsg.Content.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeImage &&
					imgMsg.Content.Parts[0].ImageUrl.Url == "data:image/jpeg;base64,<BASE64_IMAGE_DATA>"
			},
		},
		{
			name: "tool use message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleAssistant,
						Content: anthropic.MessageContents{
							{
								Type:  anthropic.MessageContentTypeToolUse,
								ID:    "tool_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location": "San Francisco"}`),
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				toolMsg := dst.Messages[0] // No system message since System field is empty
				return toolMsg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					len(toolMsg.ToolCalls) == 1 &&
					toolMsg.ToolCalls[0].ID == "tool_123" &&
					toolMsg.ToolCalls[0].Type == openrouter.ChatCompletionMessageToolCallTypeFunction &&
					toolMsg.ToolCalls[0].Function.Name == "get_weather" &&
					toolMsg.ToolCalls[0].Function.Arguments == `{"location": "San Francisco"}`
			},
		},
		{
			name: "multiple tool use message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleAssistant,
						Content: anthropic.MessageContents{
							{
								Type:  anthropic.MessageContentTypeToolUse,
								ID:    "tool_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location": "San Francisco"}`),
							},
							{
								Type:  anthropic.MessageContentTypeToolUse,
								ID:    "tool_456",
								Name:  "get_time",
								Input: json.RawMessage(`{"timezone": "UTC"}`),
							},
							{
								Type:  anthropic.MessageContentTypeToolUse,
								ID:    "tool_789",
								Name:  "calculate",
								Input: json.RawMessage(`{"expression": "2+2"}`),
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				// Multiple tool uses from the same message get merged into one message
				if len(dst.Messages) != 1 {
					return false
				}
				// Check the merged tool message
				toolMsg := dst.Messages[0]

				return toolMsg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					len(toolMsg.ToolCalls) == 3 &&
					toolMsg.ToolCalls[0].ID == "tool_123" &&
					toolMsg.ToolCalls[0].Function.Name == "get_weather" &&
					toolMsg.ToolCalls[1].ID == "tool_456" &&
					toolMsg.ToolCalls[1].Function.Name == "get_time" &&
					toolMsg.ToolCalls[2].ID == "tool_789" &&
					toolMsg.ToolCalls[2].Function.Name == "calculate"
			},
		},
		{
			name: "tool result message conversion",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type:      anthropic.MessageContentTypeToolResult,
								ToolUseID: "tool_123",
								Content: anthropic.MessageContents{
									{
										Type: anthropic.MessageContentTypeText,
										Text: "The weather is sunny, 72°F",
									},
								},
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) < 1 {
					return false
				}
				toolResultMsg := dst.Messages[0] // No system message since System field is empty
				return toolResultMsg.Role == openrouter.ChatCompletionMessageRoleTool &&
					toolResultMsg.ToolCallID == "tool_123" &&
					toolResultMsg.Content.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(toolResultMsg.Content.Parts) == 1 &&
					toolResultMsg.Content.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					toolResultMsg.Content.Parts[0].Text == "The weather is sunny, 72°F"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tt.src)
			if !tt.want(got) {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() message validation failed")
			}
		})
	}
}

func TestConvertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent(t *testing.T) {
	tests := []struct {
		name string
		src  anthropic.MessageContents
		want func(*openrouter.ChatCompletionMessageContent) bool
	}{
		{
			name: "empty content",
			src:  anthropic.MessageContents{},
			want: func(dst *openrouter.ChatCompletionMessageContent) bool {
				return dst.Type == openrouter.ChatCompletionMessageContentTypeText &&
					dst.Text == ""
			},
		},
		{
			name: "single text content",
			src: anthropic.MessageContents{
				{
					Type: anthropic.MessageContentTypeText,
					Text: "Single text result",
				},
			},
			want: func(dst *openrouter.ChatCompletionMessageContent) bool {
				return dst.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(dst.Parts) == 1 &&
					dst.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					dst.Parts[0].Text == "Single text result"
			},
		},
		{
			name: "multiple text contents",
			src: anthropic.MessageContents{
				{
					Type: anthropic.MessageContentTypeText,
					Text: "First part",
				},
				{
					Type: anthropic.MessageContentTypeText,
					Text: "Second part",
				},
			},
			want: func(dst *openrouter.ChatCompletionMessageContent) bool {
				return dst.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(dst.Parts) == 2 &&
					dst.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					dst.Parts[0].Text == "First part" &&
					dst.Parts[1].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					dst.Parts[1].Text == "Second part"
			},
		},
		{
			name: "mixed text and image content",
			src: anthropic.MessageContents{
				{
					Type: anthropic.MessageContentTypeText,
					Text: "Here's the result:",
				},
				{
					Type: anthropic.MessageContentTypeImage,
					Source: &anthropic.MessageContentSource{
						Type:      "base64",
						MediaType: "image/png",
						Data:      "<BASE64_IMAGE_DATA>",
					},
				},
			},
			want: func(dst *openrouter.ChatCompletionMessageContent) bool {
				return dst.Type == openrouter.ChatCompletionMessageContentTypeParts &&
					len(dst.Parts) == 2 &&
					dst.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					dst.Parts[0].Text == "Here's the result:" &&
					dst.Parts[1].Type == openrouter.ChatCompletionMessageContentPartTypeImage &&
					dst.Parts[1].ImageUrl.Url == "data:image/png;base64,<BASE64_IMAGE_DATA>"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent(tt.src)
			if !tt.want(got) {
				t.Errorf("convertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent() validation failed")
			}
		})
	}
}

func TestCanonicalOpenRouterMessages(t *testing.T) {
	tests := []struct {
		name string
		src  []*openrouterChatCompletionMessageWrapper
		want func([]*openrouter.ChatCompletionMessage) bool
	}{
		{
			name: "system and tool messages pass through",
			src: []*openrouterChatCompletionMessageWrapper{
				{
					ChatCompletionMessage: &openrouter.ChatCompletionMessage{
						Role: openrouter.ChatCompletionMessageRoleSystem,
						Content: &openrouter.ChatCompletionMessageContent{
							Type: openrouter.ChatCompletionMessageContentTypeText,
							Text: "System prompt",
						},
					},
				},
				{
					ChatCompletionMessage: &openrouter.ChatCompletionMessage{
						Role:       openrouter.ChatCompletionMessageRoleTool,
						ToolCallID: "tool_123",
						Content: &openrouter.ChatCompletionMessageContent{
							Type: openrouter.ChatCompletionMessageContentTypeText,
							Text: "Tool result",
						},
					},
				},
			},
			want: func(messages []*openrouter.ChatCompletionMessage) bool {
				return len(messages) == 2 &&
					messages[0].Role == openrouter.ChatCompletionMessageRoleSystem &&
					messages[0].Content.Text == "System prompt" &&
					messages[1].Role == openrouter.ChatCompletionMessageRoleTool &&
					messages[1].Content.Text == "Tool result"
			},
		},
		{
			name: "merge assistant messages with same underlying message",
			src: func() []*openrouterChatCompletionMessageWrapper {
				// Create a single anthropic message instance to be shared
				sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}
				return []*openrouterChatCompletionMessageWrapper{
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role: openrouter.ChatCompletionMessageRoleAssistant,
							Content: &openrouter.ChatCompletionMessageContent{
								Type: openrouter.ChatCompletionMessageContentTypeText,
								Text: "First part",
							},
						},
						underlyingAnthropicMessage: sharedMsg,
					},
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role:      openrouter.ChatCompletionMessageRoleAssistant,
							Reasoning: "Some thinking",
							ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
								{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "Some thinking"},
							},
						},
						underlyingAnthropicMessage: sharedMsg, // Same message reference
					},
				}
			}(),
			want: func(messages []*openrouter.ChatCompletionMessage) bool {
				if len(messages) != 1 {
					return false
				}
				msg := messages[0]
				return msg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					msg.Content.Text == "First part" &&
					msg.Reasoning == "Some thinking" &&
					len(msg.ReasoningDetails) == 1 &&
					msg.ReasoningDetails[0].Index == 0
			},
		},
		{
			name: "merge tool calls in assistant message",
			src: func() []*openrouterChatCompletionMessageWrapper {
				// Create a single anthropic message instance to be shared
				sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}
				return []*openrouterChatCompletionMessageWrapper{
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role: openrouter.ChatCompletionMessageRoleAssistant,
							Content: &openrouter.ChatCompletionMessageContent{
								Type: openrouter.ChatCompletionMessageContentTypeText,
								Text: "I'll call some tools",
							},
						},
						underlyingAnthropicMessage: sharedMsg,
					},
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role: openrouter.ChatCompletionMessageRoleAssistant,
							ToolCalls: []*openrouter.ChatCompletionToolCall{
								{
									ID:   "call1",
									Type: openrouter.ChatCompletionMessageToolCallTypeFunction,
									Function: &openrouter.ChatCompletionMessageToolCallFunction{
										Name:      "tool1",
										Arguments: "{}",
									},
								},
							},
						},
						underlyingAnthropicMessage: sharedMsg, // Same message reference
					},
				}
			}(),
			want: func(messages []*openrouter.ChatCompletionMessage) bool {
				if len(messages) != 1 {
					return false
				}
				msg := messages[0]
				return msg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					msg.Content.Text == "I'll call some tools" &&
					len(msg.ToolCalls) == 1 &&
					msg.ToolCalls[0].ID == "call1" &&
					msg.ToolCalls[0].Index == 0
			},
		},
		{
			name: "merge multiple tool calls and reasoning details",
			src: func() []*openrouterChatCompletionMessageWrapper {
				// Create a single anthropic message instance to be shared
				sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}
				return []*openrouterChatCompletionMessageWrapper{
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role: openrouter.ChatCompletionMessageRoleAssistant,
							Content: &openrouter.ChatCompletionMessageContent{
								Type: openrouter.ChatCompletionMessageContentTypeText,
								Text: "Let me use some tools and think about this",
							},
							ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
								{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "First thought"},
								{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "Second thought"},
							},
						},
						underlyingAnthropicMessage: sharedMsg,
					},
					{
						ChatCompletionMessage: &openrouter.ChatCompletionMessage{
							Role: openrouter.ChatCompletionMessageRoleAssistant,
							ToolCalls: []*openrouter.ChatCompletionToolCall{
								{
									ID:   "call1",
									Type: openrouter.ChatCompletionMessageToolCallTypeFunction,
									Function: &openrouter.ChatCompletionMessageToolCallFunction{
										Name:      "tool1",
										Arguments: "{}",
									},
								},
								{
									ID:   "call2",
									Type: openrouter.ChatCompletionMessageToolCallTypeFunction,
									Function: &openrouter.ChatCompletionMessageToolCallFunction{
										Name:      "tool2",
										Arguments: `{"param": "value"}`,
									},
								},
							},
							ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
								{Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText, Text: "Third thought"},
							},
						},
						underlyingAnthropicMessage: sharedMsg, // Same message reference
					},
				}
			}(),
			want: func(messages []*openrouter.ChatCompletionMessage) bool {
				if len(messages) != 1 {
					return false
				}
				msg := messages[0]
				return msg.Role == openrouter.ChatCompletionMessageRoleAssistant &&
					msg.Content.Text == "Let me use some tools and think about this" &&
					len(msg.ToolCalls) == 2 &&
					msg.ToolCalls[0].ID == "call1" &&
					msg.ToolCalls[0].Index == 0 &&
					msg.ToolCalls[1].ID == "call2" &&
					msg.ToolCalls[1].Index == 1 &&
					len(msg.ReasoningDetails) == 3 &&
					msg.ReasoningDetails[0].Text == "First thought" &&
					msg.ReasoningDetails[0].Index == 0 &&
					msg.ReasoningDetails[1].Text == "Second thought" &&
					msg.ReasoningDetails[1].Index == 1 &&
					msg.ReasoningDetails[2].Text == "Third thought" &&
					msg.ReasoningDetails[2].Index == 2
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := canonicalOpenRouterMessages(testProfile(), "claude-3-5-sonnet-20241022", tt.src)
			if !tt.want(got) {
				t.Errorf("canonicalOpenRouterMessages() validation failed")
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(*openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "empty system message should not create system message",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				System:    anthropic.MessageContents{}, // Empty system message
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeText,
								Text: "Hello",
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return len(dst.Messages) == 1 &&
					dst.Messages[0].Role == openrouter.ChatCompletionMessageRoleUser
			},
		},
		{
			name: "empty metadata should not set user",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Metadata:  &anthropic.Metadata{UserID: ""},
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.User == ""
			},
		},
		{
			name: "nil metadata should not set user",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return dst.User == ""
			},
		},
		{
			name: "empty tools should not panic",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Tools:     []*anthropic.Tool{},
				Messages:  []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				return len(dst.Tools) == 0
			},
		},
		{
			name: "image with nil source should not crash",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type:   anthropic.MessageContentTypeImage,
								Source: nil, // This should not cause a crash
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				// Should have no messages since image with nil source is skipped and System is empty
				return len(dst.Messages) == 0
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tt.src)
			if !tt.want(got) {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() edge case validation failed")
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_PanicScenarios(t *testing.T) {
	t.Run("redacted thinking should panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for redacted thinking")
			}
		}()

		src := &anthropic.GenerateMessageRequest{
			Model:     "claude-3-5-sonnet-20241022",
			MaxTokens: 500,
			Messages: []*anthropic.Message{
				{
					Role: anthropic.MessageRoleAssistant,
					Content: anthropic.MessageContents{
						{
							Type: anthropic.MessageContentTypeRedactedThinking,
						},
					},
				},
			},
		}

		ConvertAnthropicRequestToOpenRouterRequest(testCtx(), src)
	})
}

func TestConvertAnthropicRequestToOpenRouterRequest_ModelMapper(t *testing.T) {
	tests := []struct {
		name        string
		src         *anthropic.GenerateMessageRequest
		modelMapper map[string]string
		wantModel   string
	}{
		{
			name: "model mapped successfully",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
			},
			wantModel: "anthropic/claude-3-5-sonnet:beta",
		},
		{
			name: "model not in mapper uses original",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-opus-20240229",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
			},
			wantModel: "claude-3-opus-20240229",
		},
		{
			name: "empty mapper uses original model",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: map[string]string{},
			wantModel:   "claude-3-5-sonnet-20241022",
		},
		{
			name: "nil mapper uses original model",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: nil,
			wantModel:   "claude-3-5-sonnet-20241022",
		},
		{
			name: "multiple model mappings",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-haiku-20240307",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
				"claude-3-opus-20240229":     "anthropic/claude-3-opus:beta",
				"claude-3-haiku-20240307":    "anthropic/claude-3-haiku:beta",
			},
			wantModel: "anthropic/claude-3-haiku:beta",
		},
		{
			name: "case sensitive mapping",
			src: &anthropic.GenerateMessageRequest{
				Model:     "Claude-3-5-Sonnet-20241022",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			modelMapper: map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
			},
			wantModel: "Claude-3-5-Sonnet-20241022", // Should not match due to case sensitivity
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := testCtxWithOptions(func(p *profile.Profile) {
				p.Options.Models = tt.modelMapper
			})

			got := ConvertAnthropicRequestToOpenRouterRequest(ctx, tt.src)

			if got.Model != tt.wantModel {
				t.Errorf("ConvertAnthropicRequestToOpenRouterRequest() model = %q, want %q", got.Model, tt.wantModel)
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ModelMapperIntegration(t *testing.T) {
	// Test that model mapper works alongside other features
	t.Run("model mapper with metadata and tools", func(t *testing.T) {
		src := &anthropic.GenerateMessageRequest{
			Model:     "claude-3-5-sonnet-20241022",
			MaxTokens: 1000,
			Metadata: &anthropic.Metadata{
				UserID: "user456",
			},
			Tools: []*anthropic.Tool{
				{
					Type:        lo.ToPtr(anthropic.ToolTypeCustom),
					Name:        "get_weather",
					Description: "Get weather info",
					InputSchema: json.RawMessage(`{
						"type": "object",
						"properties": {
							"location": {
								"type": "string"
							}
						}
					}`),
				},
			},
			Messages: []*anthropic.Message{
				{
					Role: anthropic.MessageRoleUser,
					Content: anthropic.MessageContents{
						{
							Type: anthropic.MessageContentTypeText,
							Text: "What's the weather like?",
						},
					},
				},
			},
		}

		ctx := testCtxWithOptions(func(p *profile.Profile) {
			p.Options.Models = map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
			}
		})

		got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

		// Verify model was mapped
		if got.Model != "anthropic/claude-3-5-sonnet:beta" {
			t.Errorf("Expected model to be mapped to 'anthropic/claude-3-5-sonnet:beta', got %q", got.Model)
		}

		// Verify other features still work
		if got.User != "user456" {
			t.Errorf("Expected user to be 'user456', got %q", got.User)
		}

		if len(got.Tools) != 1 {
			t.Errorf("Expected 1 tool, got %d", len(got.Tools))
		}

		if len(got.Messages) != 1 {
			t.Errorf("Expected 1 message, got %d", len(got.Messages))
		}
	})

	// Test multiple options combined
	t.Run("multiple mapper mappings with complex request", func(t *testing.T) {
		src := &anthropic.GenerateMessageRequest{
			Model:         "claude-3-haiku-20240307",
			MaxTokens:     500,
			Temperature:   0.5,
			StopSequences: []string{"END"},
			Messages: []*anthropic.Message{
				{
					Role: anthropic.MessageRoleUser,
					Content: anthropic.MessageContents{
						{
							Type: anthropic.MessageContentTypeText,
							Text: "Hello",
						},
					},
				},
			},
		}

		ctx := testCtxWithOptions(func(p *profile.Profile) {
			p.Options.Models = map[string]string{
				"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet:beta",
				"claude-3-opus-20240229":     "anthropic/claude-3-opus:beta",
				"claude-3-haiku-20240307":    "anthropic/claude-3-haiku:beta",
			}
		})

		got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

		if got.Model != "anthropic/claude-3-haiku:beta" {
			t.Errorf("Expected model 'anthropic/claude-3-haiku:beta', got %q", got.Model)
		}

		if *got.Temperature != 0.5 {
			t.Errorf("Expected temperature 0.5, got %v", *got.Temperature)
		}

		if len(got.Stop) != 1 || got.Stop[0] != "END" {
			t.Errorf("Expected stop sequence ['END'], got %v", got.Stop)
		}
	})
}

func TestConvertAnthropicRequestToOpenRouterRequest_CacheControl(t *testing.T) {
	testCases := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(dst *openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "system message with cache control",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				System: anthropic.MessageContents{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "You are a helpful assistant",
						CacheControl: &anthropic.CacheControl{
							Type: anthropic.MessageCacheControlTypeEphemeral,
							TTL:  anthropic.MessageCacheControlTTL5Minutes,
						},
					},
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) != 1 {
					return false
				}
				msg := dst.Messages[0]
				if msg.Role != openrouter.ChatCompletionMessageRoleSystem {
					return false
				}
				if !msg.Content.IsParts() || len(msg.Content.Parts) != 1 {
					return false
				}
				part := msg.Content.Parts[0]
				return part.CacheControl != nil &&
					string(part.CacheControl.Type) == string(anthropic.MessageCacheControlTypeEphemeral) &&
					string(part.CacheControl.TTL) == string(anthropic.MessageCacheControlTTL5Minutes)
			},
		},
		{
			name: "user text message with cache control",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeText,
								Text: "Hello with cache",
								CacheControl: &anthropic.CacheControl{
									Type: anthropic.MessageCacheControlTypeEphemeral,
									TTL:  anthropic.MessageCacheControlTTL1Hour,
								},
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) != 1 {
					return false
				}
				msg := dst.Messages[0]
				if !msg.Content.IsParts() || len(msg.Content.Parts) != 1 {
					return false
				}
				part := msg.Content.Parts[0]
				return part.CacheControl != nil &&
					string(part.CacheControl.Type) == string(anthropic.MessageCacheControlTypeEphemeral) &&
					string(part.CacheControl.TTL) == string(anthropic.MessageCacheControlTTL1Hour)
			},
		},
		{
			name: "user image message with cache control",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeImage,
								Source: &anthropic.MessageContentSource{
									Type:      anthropic.MessageContentTypeImage,
									MediaType: "image/png",
									Data:      "<BASE64_IMAGE_DATA>",
								},
								CacheControl: &anthropic.CacheControl{
									Type: anthropic.MessageCacheControlTypeEphemeral,
								},
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) != 1 {
					return false
				}
				msg := dst.Messages[0]
				if !msg.Content.IsParts() || len(msg.Content.Parts) != 1 {
					return false
				}
				part := msg.Content.Parts[0]
				// Cache control should be nil due to cleanup logic - OpenRouter only allows text parts to have CacheControl
				return part.CacheControl == nil
			},
		},
		{
			name: "tool result with cache control",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type:      anthropic.MessageContentTypeToolResult,
								ToolUseID: "tool_123",
								Content: anthropic.MessageContents{
									{
										Type: anthropic.MessageContentTypeText,
										Text: "Tool result text",
										CacheControl: &anthropic.CacheControl{
											Type: anthropic.MessageCacheControlTypeEphemeral,
										},
									},
								},
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) != 1 {
					return false
				}
				msg := dst.Messages[0]
				if msg.Role != openrouter.ChatCompletionMessageRoleTool {
					return false
				}
				if !msg.Content.IsParts() || len(msg.Content.Parts) != 1 {
					return false
				}
				part := msg.Content.Parts[0]
				return part.CacheControl != nil &&
					string(part.CacheControl.Type) == string(anthropic.MessageCacheControlTypeEphemeral)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tc.src)
			if !tc.want(got) {
				t.Errorf("Test case %s failed validation", tc.name)
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_SystemMessageTypes(t *testing.T) {
	testCases := []struct {
		name string
		src  *anthropic.GenerateMessageRequest
		want func(dst *openrouter.CreateChatCompletionRequest) bool
	}{
		{
			name: "system message with text and image",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				System: anthropic.MessageContents{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "You are a helpful assistant",
					},
					{
						Type: anthropic.MessageContentTypeImage,
						Source: &anthropic.MessageContentSource{
							Type:      anthropic.MessageContentTypeImage,
							MediaType: "image/png",
							Data:      "<BASE64_IMAGE_DATA>",
						},
					},
				},
				Messages: []*anthropic.Message{},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				if len(dst.Messages) != 1 {
					return false
				}
				msg := dst.Messages[0]
				if msg.Role != openrouter.ChatCompletionMessageRoleSystem {
					return false
				}
				if !msg.Content.IsParts() || len(msg.Content.Parts) != 2 {
					return false
				}
				return msg.Content.Parts[0].Type == openrouter.ChatCompletionMessageContentPartTypeText &&
					msg.Content.Parts[0].Text == "You are a helpful assistant" &&
					msg.Content.Parts[1].Type == openrouter.ChatCompletionMessageContentPartTypeImage &&
					msg.Content.Parts[1].ImageUrl != nil
			},
		},
		{
			name: "empty system message should not create system message",
			src: &anthropic.GenerateMessageRequest{
				Model:     "claude-3-5-sonnet-20241022",
				MaxTokens: 500,
				System:    anthropic.MessageContents{},
				Messages: []*anthropic.Message{
					{
						Role: anthropic.MessageRoleUser,
						Content: anthropic.MessageContents{
							{
								Type: anthropic.MessageContentTypeText,
								Text: "Hello",
							},
						},
					},
				},
			},
			want: func(dst *openrouter.CreateChatCompletionRequest) bool {
				// Should only have the user message, no system message
				return len(dst.Messages) == 1 && dst.Messages[0].Role == openrouter.ChatCompletionMessageRoleUser
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), tc.src)
			if !tc.want(got) {
				t.Errorf("Test case %s failed validation", tc.name)
			}
		})
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ToolTypeNil(t *testing.T) {
	// Test that a tool with nil Type is treated as custom tool
	ctx := testCtxWithOptions(func(p *profile.Profile) {
		p.Options.Strict = true
	})

	inputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"param": map[string]any{
				"type": "string",
			},
		},
	}
	inputSchemaBytes, _ := json.Marshal(inputSchema)

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Tools: []*anthropic.Tool{
			{
				Type:        nil, // nil Type should be treated as custom tool
				Name:        "custom_tool_with_nil_type",
				Description: "Custom tool without explicit type",
				InputSchema: json.RawMessage(inputSchemaBytes),
			},
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	if len(got.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(got.Tools))
		return
	}

	tool := got.Tools[0]
	if tool.Type != openrouter.ChatCompletionMessageToolCallTypeFunction {
		t.Errorf("Expected tool type function, got %s", tool.Type)
	}

	if tool.Function.Name != "custom_tool_with_nil_type" {
		t.Errorf("Expected tool name custom_tool_with_nil_type, got %s", tool.Function.Name)
	}

	if tool.Function.Description != "Custom tool without explicit type" {
		t.Errorf("Expected tool description 'Custom tool without explicit type', got %s", tool.Function.Description)
	}

	if !tool.Function.Strict {
		t.Error("Expected strict mode to be true")
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_WebSearchTool(t *testing.T) {
	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Tools: []*anthropic.Tool{
			{
				Type:           lo.ToPtr(anthropic.ToolTypeWebSearch2025),
				Name:           "web_search",
				MaxUses:        5,
				AllowedDomains: []string{"example.com", "wikipedia.org"},
				BlockedDomains: []string{"blocked.com"},
				UserLocation: &anthropic.ToolLocation{
					Type:     anthropic.ToolLocationTypeApproximate,
					City:     "San Francisco",
					Region:   "California",
					Country:  "US",
					Timezone: "America/Los_Angeles",
				},
			},
			{
				Type:        lo.ToPtr(anthropic.ToolTypeCustom),
				Name:        "custom_tool",
				Description: "Custom tool",
				InputSchema: json.RawMessage(`{"type": "object"}`),
				CacheControl: &anthropic.CacheControl{
					Type: anthropic.MessageCacheControlTypeEphemeral,
				},
			},
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), src)

	// Web Search Tool should be filtered out (not ToolTypeCustom)
	if len(got.Tools) != 1 {
		t.Errorf("Expected 1 tool (custom only), got %d", len(got.Tools))
	}

	if got.Tools[0].Function.Name != "custom_tool" {
		t.Errorf("Expected custom_tool, got %s", got.Tools[0].Function.Name)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_CacheControlCleanup(t *testing.T) {
	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Text with cache",
						CacheControl: &anthropic.CacheControl{
							Type: anthropic.MessageCacheControlTypeEphemeral,
						},
					},
					{
						Type: anthropic.MessageContentTypeImage,
						Source: &anthropic.MessageContentSource{
							Type:      anthropic.MessageContentTypeImage,
							MediaType: "image/png",
							Data:      "<BASE64_IMAGE_DATA>",
						},
						CacheControl: &anthropic.CacheControl{
							Type: anthropic.MessageCacheControlTypeEphemeral,
						},
					},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), src)

	// Should have 1 message with 2 parts (merged by canonicalOpenRouterMessages)
	if len(got.Messages) != 1 {
		t.Errorf("Expected 1 message (merged), got %d", len(got.Messages))
		return
	}

	msg := got.Messages[0]
	if !msg.Content.IsParts() || len(msg.Content.Parts) != 2 {
		t.Errorf("Message should have 2 parts, got %d", len(msg.Content.Parts))
		return
	}

	// Check text part retains cache control
	textPart := msg.Content.Parts[0]
	if textPart.Type != openrouter.ChatCompletionMessageContentPartTypeText {
		t.Error("First part should be text")
		return
	}
	if textPart.CacheControl == nil {
		t.Error("Text part should retain cache control")
	}

	// Check image part loses cache control (due to cleanup logic)
	imgPart := msg.Content.Parts[1]
	if imgPart.Type != openrouter.ChatCompletionMessageContentPartTypeImage {
		t.Error("Second part should be image")
		return
	}
	if imgPart.CacheControl != nil {
		t.Error("Image part cache control should be cleaned up")
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_NewMessageFormat(t *testing.T) {
	// Test that all messages now use Parts format instead of simple text
	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Hello",
					},
				},
			},
			{
				Role: anthropic.MessageRoleAssistant,
				Content: anthropic.MessageContents{
					{
						Type: anthropic.MessageContentTypeText,
						Text: "Hi there",
					},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(testCtx(), src)

	if len(got.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(got.Messages))
		return
	}

	// Check user message (should use Parts format)
	userMsg := got.Messages[0]
	if !userMsg.Content.IsParts() {
		t.Error("User message should use Parts format")
	}
	if len(userMsg.Content.Parts) != 1 {
		t.Errorf("User message should have 1 part, got %d", len(userMsg.Content.Parts))
	}
	if userMsg.Content.Parts[0].Type != openrouter.ChatCompletionMessageContentPartTypeText {
		t.Error("User message part should be text type")
	}

	// Check assistant message (should be converted to text format)
	assistantMsg := got.Messages[1]
	if !assistantMsg.Content.IsText() {
		t.Error("Assistant message should use Text format (converted from Parts)")
	}
	if assistantMsg.Content.Text != "Hi there" {
		t.Errorf("Assistant message text should be 'Hi there', got '%s'", assistantMsg.Content.Text)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ReasoningFormat_AnthropicClaudeV1(t *testing.T) {
	ctx := testCtxWithReasoningFormat("anthropic-claude-v1", "high")

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 123,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Reasoning.Effort != "" {
		t.Errorf("Effort should be cleared, got %q", got.Reasoning.Effort)
	}
	if got.Reasoning.MaxTokens != 123 {
		t.Errorf("MaxTokens should remain 123, got %d", got.Reasoning.MaxTokens)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ReasoningFormat_OpenAIResponsesV1(t *testing.T) {
	ctx := testCtxWithReasoningFormat("openai-responses-v1", "medium")

	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-5",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 200,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Reasoning.MaxTokens != 0 {
		t.Errorf("MaxTokens should be zeroed for OpenAIResponsesV1, got %d", got.Reasoning.MaxTokens)
	}
	if string(got.Reasoning.Effort) != "medium" {
		t.Errorf("Effort should retain configured value, got %q", got.Reasoning.Effort)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_OpenAIResponsesV1_ModelEffortSuffix(t *testing.T) {
	ctx := testCtxWithReasoningFormat("openai-responses-v1", "low")

	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-5:high",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 200,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Model != "gpt-5" {
		t.Errorf("Model should drop suffix, got %q", got.Model)
	}
	if got.Reasoning.MaxTokens != 0 {
		t.Errorf("MaxTokens should be zeroed, got %d", got.Reasoning.MaxTokens)
	}
	if string(got.Reasoning.Effort) != "high" {
		t.Errorf("Effort should be parsed from suffix, got %q", got.Reasoning.Effort)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_OpenAIResponsesV1_NoSuffixKeepsConfigEffort(t *testing.T) {
	ctx := testCtxWithReasoningFormat("openai-responses-v1", "low")

	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-5",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 200,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Model != "gpt-5" {
		t.Errorf("Model should remain unchanged, got %q", got.Model)
	}
	if got.Reasoning.MaxTokens != 0 {
		t.Errorf("MaxTokens should be zeroed, got %d", got.Reasoning.MaxTokens)
	}
	if string(got.Reasoning.Effort) != "low" {
		t.Errorf("Effort should use configured value, got %q", got.Reasoning.Effort)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_AnthropicClaudeV1_IgnoreSuffixAndKeepModel(t *testing.T) {
	ctx := testCtxWithReasoningFormat("anthropic-claude-v1", "high")

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-7-sonnet-20250219:thinking",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 123,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Model != "claude-3-7-sonnet-20250219:thinking" {
		t.Errorf("Model should not be split in AnthropicClaudeV1, got %q", got.Model)
	}
	if got.Reasoning.Effort != "" {
		t.Errorf("Effort should be cleared, got %q", got.Reasoning.Effort)
	}
	if got.Reasoning.MaxTokens != 123 {
		t.Errorf("MaxTokens should remain 123, got %d", got.Reasoning.MaxTokens)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_OpenAIResponsesV1_ModelEffortSuffix_NoThinking(t *testing.T) {
	ctx := testCtxWithReasoningFormat("openai-responses-v1", "low")

	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-5:high",
		MaxTokens: 256,
		Messages:  []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	if got.Model != "gpt-5" {
		t.Errorf("Model should drop suffix, got %q", got.Model)
	}
	if got.Reasoning.MaxTokens != 0 {
		t.Errorf("MaxTokens should be zeroed, got %d", got.Reasoning.MaxTokens)
	}
	if string(got.Reasoning.Effort) != "high" {
		t.Errorf("Effort should be parsed from suffix, got %q", got.Reasoning.Effort)
	}
}

func TestDefaultReasoningEffort(t *testing.T) {
	// Default profile has empty effort
	p := testProfile()
	if effort := openrouter.ChatCompletionReasoningEffort(p.Options.Reasoning.Effort); !effort.IsEmpty() {
		t.Errorf("Effort should be empty, got %q", effort)
	}
}

func TestForceThinking_AnthropicClaudeV1_NoReasoningAddsReasoning(t *testing.T) {
	ctx := testCtxWithForceThinking("anthropic-claude-v1")

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 0,
		Messages:  []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	if got.Reasoning == nil || !got.Reasoning.Enabled {
		t.Fatalf("force thinking should enable reasoning")
	}
	if got.MaxTokens == nil || *got.MaxTokens != 32*1024 {
		t.Errorf("MaxTokens should be set to 32768, got %v", lo.FromPtr(got.MaxTokens))
	}
	if got.Reasoning.MaxTokens != 32*1024-1 {
		t.Errorf("Reasoning.MaxTokens should be 32767, got %d", got.Reasoning.MaxTokens)
	}
}

func TestForceThinking_DoesNotOverrideExistingReasoning(t *testing.T) {
	ctx := testCtxWithForceThinking("anthropic-claude-v1")

	src := &anthropic.GenerateMessageRequest{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: 1000,
		Thinking:  &anthropic.Thinking{Type: anthropic.ThinkingTypeEnabled, BudgetTokens: 1234},
		Messages:  []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	if got.Reasoning == nil || !got.Reasoning.Enabled {
		t.Fatalf("Reasoning should remain enabled from source")
	}
	if got.Reasoning.MaxTokens != 1234 {
		t.Errorf("Existing Reasoning.MaxTokens should be preserved, got %d", got.Reasoning.MaxTokens)
	}
}

func TestForceThinking_MaxTokensBoundaries(t *testing.T) {
	ctx := testCtxWithForceThinking("anthropic-claude-v1")

	srcSmall := &anthropic.GenerateMessageRequest{Model: "claude-3-5-sonnet-20241022", MaxTokens: 10, Messages: []*anthropic.Message{}}
	gotSmall := ConvertAnthropicRequestToOpenRouterRequest(ctx, srcSmall)
	if gotSmall.Reasoning == nil || !gotSmall.Reasoning.Enabled {
		t.Fatalf("force thinking should enable reasoning (small)")
	}
	if gotSmall.MaxTokens == nil || *gotSmall.MaxTokens != 32*1024 {
		t.Errorf("MaxTokens should be promoted to 32768 when <=1024, got %v", lo.FromPtr(gotSmall.MaxTokens))
	}
	if gotSmall.Reasoning.MaxTokens != 32*1024-1 {
		t.Errorf("Reasoning.MaxTokens should be 32767, got %d", gotSmall.Reasoning.MaxTokens)
	}

	srcLarge := &anthropic.GenerateMessageRequest{Model: "claude-3-5-sonnet-20241022", MaxTokens: 100000, Messages: []*anthropic.Message{}}
	gotLarge := ConvertAnthropicRequestToOpenRouterRequest(ctx, srcLarge)
	if gotLarge.Reasoning == nil || !gotLarge.Reasoning.Enabled {
		t.Fatalf("force thinking should enable reasoning (large)")
	}
	if gotLarge.MaxTokens == nil || *gotLarge.MaxTokens != 100000 {
		t.Errorf("MaxTokens should remain 100000, got %v", lo.FromPtr(gotLarge.MaxTokens))
	}
	if gotLarge.Reasoning.MaxTokens != 99999 {
		t.Errorf("Reasoning.MaxTokens should be MaxTokens-1 (99999), got %d", gotLarge.Reasoning.MaxTokens)
	}
	if gotLarge.MaxTokens == nil || *gotLarge.MaxTokens != 100000 {
		t.Errorf("Original MaxTokens should remain 100000, got %v", lo.FromPtr(gotLarge.MaxTokens))
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_ReasoningFormat_GoogleGeminiV1(t *testing.T) {
	ctx := testCtxWithReasoningFormat("google-gemini-v1", "")

	src := &anthropic.GenerateMessageRequest{
		Model:     "google/gemini-3-flash-thinking",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 123,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	// Google Gemini format forces Enabled to true
	if !got.Reasoning.Enabled {
		t.Errorf("Reasoning.Enabled should be forced to true for Google Gemini, got false")
	}
	// MaxTokens should be preserved from the thinking budget
	if got.Reasoning.MaxTokens != 123 {
		t.Errorf("MaxTokens should remain 123, got %d", got.Reasoning.MaxTokens)
	}
	// Effort should be empty for Google Gemini format
	if got.Reasoning.Effort != "" {
		t.Errorf("Effort should be empty for Google Gemini, got %q", got.Reasoning.Effort)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_GoogleGeminiV1_ForceEnabled(t *testing.T) {
	ctx := testCtxWithReasoningFormat("google-gemini-v1", "")

	tests := []struct {
		name      string
		src       *anthropic.GenerateMessageRequest
		wantCheck func(*testing.T, *openrouter.CreateChatCompletionRequest)
	}{
		{
			name: "nil reasoning is created with enabled=true",
			src: &anthropic.GenerateMessageRequest{
				Model:     "google/gemini-3-flash-thinking",
				MaxTokens: 500,
				Messages:  []*anthropic.Message{},
			},
			wantCheck: func(t *testing.T, got *openrouter.CreateChatCompletionRequest) {
				if got.Reasoning == nil {
					t.Fatalf("Reasoning should be created, got nil")
				}
				if !got.Reasoning.Enabled {
					t.Errorf("Reasoning.Enabled should be forced to true, got false")
				}
			},
		},
		{
			name: "disabled reasoning is forced to enabled",
			src: &anthropic.GenerateMessageRequest{
				Model:     "google/gemini-3-flash-thinking",
				MaxTokens: 500,
				Thinking: &anthropic.Thinking{
					Type:         anthropic.ThinkingTypeDisabled,
					BudgetTokens: 100,
				},
				Messages: []*anthropic.Message{},
			},
			wantCheck: func(t *testing.T, got *openrouter.CreateChatCompletionRequest) {
				if got.Reasoning == nil {
					t.Fatalf("Reasoning should exist, got nil")
				}
				if !got.Reasoning.Enabled {
					t.Errorf("Reasoning.Enabled should be forced to true even when source is disabled, got false")
				}
				if got.Reasoning.MaxTokens != 100 {
					t.Errorf("MaxTokens should still be set from source, got %d", got.Reasoning.MaxTokens)
				}
			},
		},
		{
			name: "enabled reasoning remains enabled",
			src: &anthropic.GenerateMessageRequest{
				Model:     "google/gemini-3-flash-thinking",
				MaxTokens: 500,
				Thinking: &anthropic.Thinking{
					Type:         anthropic.ThinkingTypeEnabled,
					BudgetTokens: 200,
				},
				Messages: []*anthropic.Message{},
			},
			wantCheck: func(t *testing.T, got *openrouter.CreateChatCompletionRequest) {
				if got.Reasoning == nil {
					t.Fatalf("Reasoning should exist, got nil")
				}
				if !got.Reasoning.Enabled {
					t.Errorf("Reasoning.Enabled should be true, got false")
				}
				if got.Reasoning.MaxTokens != 200 {
					t.Errorf("MaxTokens should be 200, got %d", got.Reasoning.MaxTokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertAnthropicRequestToOpenRouterRequest(ctx, tt.src)
			tt.wantCheck(t, got)
		})
	}
}

func TestCanonicalOpenRouterMessages_GoogleGeminiV1Format(t *testing.T) {
	prof := testProfileWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = "google-gemini-v1"
		p.Options.Reasoning.Delimiter = "/"
	})

	// Create a single anthropic message instance to be shared
	sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}

	src := []*openrouterChatCompletionMessageWrapper{
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Let me think about this",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text:      "First thought",
						Signature: "sig123/data456",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg,
		},
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "And here's my answer",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text:      "Second thought",
						Signature: "sig789/data012",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg, // Same message reference
		},
	}

	messages := canonicalOpenRouterMessages(prof, "google/gemini-3-flash-thinking", src)
	if len(messages) != 1 {
		t.Fatalf("Expected 1 merged message, got %d", len(messages))
	}

	msg := messages[0]
	if msg.Role != openrouter.ChatCompletionMessageRoleAssistant {
		t.Errorf("Expected role assistant, got %s", msg.Role)
	}

	// Check that content is merged into Parts (two text contents -> Parts type)
	if msg.Content.Type != openrouter.ChatCompletionMessageContentTypeParts {
		t.Errorf("Expected content type parts, got %s", msg.Content.Type)
	}
	if len(msg.Content.Parts) != 2 {
		t.Fatalf("Expected 2 content parts, got %d", len(msg.Content.Parts))
	}
	if msg.Content.Parts[0].Text != "Let me think about this" {
		t.Errorf("Expected first part text 'Let me think about this', got %q", msg.Content.Parts[0].Text)
	}
	if msg.Content.Parts[1].Text != "And here's my answer" {
		t.Errorf("Expected second part text 'And here's my answer', got %q", msg.Content.Parts[1].Text)
	}

	// Check that reasoning details are properly formatted with google-gemini-v1
	if len(msg.ReasoningDetails) != 4 {
		t.Fatalf("Expected 4 reasoning details (2 summaries + 2 encrypted), got %d", len(msg.ReasoningDetails))
	}

	// First reasoning detail should be reasoning.text type with google-gemini-v1 format
	if msg.ReasoningDetails[0].Type != openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText {
		t.Errorf("Expected first reasoning detail to be reasoning.text type, got %s", msg.ReasoningDetails[0].Type)
	}
	if msg.ReasoningDetails[0].Format != openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1 {
		t.Errorf("Expected format google-gemini-v1, got %s", msg.ReasoningDetails[0].Format)
	}
	if msg.ReasoningDetails[0].Text != "First thought" {
		t.Errorf("Expected text 'First thought', got %q", msg.ReasoningDetails[0].Text)
	}

	// Second reasoning detail should be encrypted type with signature split
	if msg.ReasoningDetails[1].Type != openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted {
		t.Errorf("Expected second reasoning detail to be encrypted type, got %s", msg.ReasoningDetails[1].Type)
	}
	if msg.ReasoningDetails[1].Format != openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1 {
		t.Errorf("Expected format google-gemini-v1, got %s", msg.ReasoningDetails[1].Format)
	}
	if msg.ReasoningDetails[1].ID != "sig123" {
		t.Errorf("Expected ID 'sig123', got %q", msg.ReasoningDetails[1].ID)
	}
	if msg.ReasoningDetails[1].Data != "data456" {
		t.Errorf("Expected Data 'data456', got %q", msg.ReasoningDetails[1].Data)
	}

	// Third reasoning detail should be reasoning.text for second thought
	if msg.ReasoningDetails[2].Type != openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText {
		t.Errorf("Expected third reasoning detail to be reasoning.text type, got %s", msg.ReasoningDetails[2].Type)
	}
	if msg.ReasoningDetails[2].Format != openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1 {
		t.Errorf("Expected format google-gemini-v1, got %s", msg.ReasoningDetails[2].Format)
	}
	if msg.ReasoningDetails[2].Text != "Second thought" {
		t.Errorf("Expected text 'Second thought', got %q", msg.ReasoningDetails[2].Text)
	}

	// Fourth reasoning detail should be encrypted for second signature
	if msg.ReasoningDetails[3].Type != openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted {
		t.Errorf("Expected fourth reasoning detail to be encrypted type, got %s", msg.ReasoningDetails[3].Type)
	}
	if msg.ReasoningDetails[3].Format != openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1 {
		t.Errorf("Expected format google-gemini-v1, got %s", msg.ReasoningDetails[3].Format)
	}
	if msg.ReasoningDetails[3].ID != "sig789" {
		t.Errorf("Expected ID 'sig789', got %q", msg.ReasoningDetails[3].ID)
	}
	if msg.ReasoningDetails[3].Data != "data012" {
		t.Errorf("Expected Data 'data012', got %q", msg.ReasoningDetails[3].Data)
	}

	// Check indices are set correctly
	// Each original reasoning detail produces two entries (summary + encrypted) with the same index
	// First pair: index 0, Second pair: index 1
	if msg.ReasoningDetails[0].Index != 0 {
		t.Errorf("Expected first reasoning detail to have index 0, got %d", msg.ReasoningDetails[0].Index)
	}
	if msg.ReasoningDetails[1].Index != 0 {
		t.Errorf("Expected second reasoning detail to have index 0 (same as summary), got %d", msg.ReasoningDetails[1].Index)
	}
	if msg.ReasoningDetails[2].Index != 1 {
		t.Errorf("Expected third reasoning detail to have index 1, got %d", msg.ReasoningDetails[2].Index)
	}
	if msg.ReasoningDetails[3].Index != 1 {
		t.Errorf("Expected fourth reasoning detail to have index 1 (same as summary), got %d", msg.ReasoningDetails[3].Index)
	}
}

func TestCanonicalOpenRouterMessages_OpenAIResponsesV1Format(t *testing.T) {
	prof := testProfileWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = "openai-responses-v1"
	})

	// Create a single anthropic message instance to be shared
	sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}

	src := []*openrouterChatCompletionMessageWrapper{
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Let me think about this",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type: openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text: "First thought",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg,
		},
	}

	messages := canonicalOpenRouterMessages(prof, "gpt-5", src)
	if len(messages) != 1 {
		t.Fatalf("Expected 1 merged message, got %d", len(messages))
	}

	msg := messages[0]

	if len(msg.ReasoningDetails) != 1 {
		t.Fatalf("Expected 1 reasoning detail, got %d", len(msg.ReasoningDetails))
	}

	// For openai-responses-v1, text should be moved to summary
	if msg.ReasoningDetails[0].Type != openrouter.ChatCompletionMessageReasoningDetailTypeSummary {
		t.Errorf("Expected reasoning detail to be summary type, got %s", msg.ReasoningDetails[0].Type)
	}
	if msg.ReasoningDetails[0].Format != openrouter.ChatCompletionMessageReasoningDetailFormatOpenAIResponsesV1 {
		t.Errorf("Expected format openai-responses-v1, got %s", msg.ReasoningDetails[0].Format)
	}
	if msg.ReasoningDetails[0].Summary != "First thought" {
		t.Errorf("Expected summary 'First thought', got %q", msg.ReasoningDetails[0].Summary)
	}
	if msg.ReasoningDetails[0].Text != "" {
		t.Errorf("Expected text to be empty, got %q", msg.ReasoningDetails[0].Text)
	}
}

// Tests for ChatCompletionMessageReasoningDetailFormatUnknown

func TestConvertAnthropicRequestToOpenRouterRequest_ReasoningFormat_Unknown(t *testing.T) {
	ctx := testCtxWithReasoningFormat("unknown", "high")

	src := &anthropic.GenerateMessageRequest{
		Model:     "some-unknown-model",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 123,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	// Unknown format should behave like AnthropicClaudeV1: keep MaxTokens and clear Effort
	if got.Reasoning.Effort != "" {
		t.Errorf("Effort should be cleared for unknown format, got %q", got.Reasoning.Effort)
	}
	if got.Reasoning.MaxTokens != 123 {
		t.Errorf("MaxTokens should remain 123, got %d", got.Reasoning.MaxTokens)
	}
}

func TestConvertAnthropicRequestToOpenRouterRequest_Unknown_IgnoreSuffixAndKeepModel(t *testing.T) {
	ctx := testCtxWithReasoningFormat("unknown", "high")

	src := &anthropic.GenerateMessageRequest{
		Model:     "some-model:thinking",
		MaxTokens: 500,
		Thinking: &anthropic.Thinking{
			Type:         anthropic.ThinkingTypeEnabled,
			BudgetTokens: 123,
		},
		Messages: []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)
	if got.Reasoning == nil {
		t.Fatalf("Reasoning is nil")
	}
	// Unknown format behaves like AnthropicClaudeV1: model suffix should NOT be split
	if got.Model != "some-model:thinking" {
		t.Errorf("Model should not be split in unknown format, got %q", got.Model)
	}
	if got.Reasoning.Effort != "" {
		t.Errorf("Effort should be cleared, got %q", got.Reasoning.Effort)
	}
	if got.Reasoning.MaxTokens != 123 {
		t.Errorf("MaxTokens should remain 123, got %d", got.Reasoning.MaxTokens)
	}
}

func TestForceThinking_UnknownFormat_NoReasoningAddsReasoning(t *testing.T) {
	ctx := testCtxWithForceThinking("unknown")

	src := &anthropic.GenerateMessageRequest{
		Model:     "some-unknown-model",
		MaxTokens: 0,
		Messages:  []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	// Unknown format should behave like AnthropicClaudeV1 with force thinking
	if got.Reasoning == nil || !got.Reasoning.Enabled {
		t.Fatalf("force thinking should enable reasoning for unknown format")
	}
	if got.MaxTokens == nil || *got.MaxTokens != 32*1024 {
		t.Errorf("MaxTokens should be set to 32768, got %v", lo.FromPtr(got.MaxTokens))
	}
	if got.Reasoning.MaxTokens != 32*1024-1 {
		t.Errorf("Reasoning.MaxTokens should be 32767, got %d", got.Reasoning.MaxTokens)
	}
}

func TestForceThinking_UnknownFormat_DoesNotOverrideExistingReasoning(t *testing.T) {
	ctx := testCtxWithForceThinking("unknown")

	src := &anthropic.GenerateMessageRequest{
		Model:     "some-unknown-model",
		MaxTokens: 1000,
		Thinking:  &anthropic.Thinking{Type: anthropic.ThinkingTypeEnabled, BudgetTokens: 1234},
		Messages:  []*anthropic.Message{},
	}

	got := ConvertAnthropicRequestToOpenRouterRequest(ctx, src)

	if got.Reasoning == nil || !got.Reasoning.Enabled {
		t.Fatalf("Reasoning should remain enabled from source")
	}
	if got.Reasoning.MaxTokens != 1234 {
		t.Errorf("Existing Reasoning.MaxTokens should be preserved, got %d", got.Reasoning.MaxTokens)
	}
}

func TestCanonicalOpenRouterMessages_UnknownFormat(t *testing.T) {
	prof := testProfileWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = "unknown"
		p.Options.Reasoning.Delimiter = "/"
	})

	// Create a single anthropic message instance to be shared
	sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}

	src := []*openrouterChatCompletionMessageWrapper{
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Let me think about this",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text:      "First thought",
						Signature: "sig123/data456",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg,
		},
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "And here's my answer",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text:      "Second thought",
						Signature: "sig789/data012",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg, // Same message reference
		},
	}

	messages := canonicalOpenRouterMessages(prof, "some-unknown-model", src)
	if len(messages) != 1 {
		t.Fatalf("Expected 1 merged message, got %d", len(messages))
	}

	msg := messages[0]
	if msg.Role != openrouter.ChatCompletionMessageRoleAssistant {
		t.Errorf("Expected role assistant, got %s", msg.Role)
	}

	// Check that content is merged into Parts
	if msg.Content.Type != openrouter.ChatCompletionMessageContentTypeParts {
		t.Errorf("Expected content type parts, got %s", msg.Content.Type)
	}
	if len(msg.Content.Parts) != 2 {
		t.Fatalf("Expected 2 content parts, got %d", len(msg.Content.Parts))
	}
	if msg.Content.Parts[0].Text != "Let me think about this" {
		t.Errorf("Expected first part text 'Let me think about this', got %q", msg.Content.Parts[0].Text)
	}
	if msg.Content.Parts[1].Text != "And here's my answer" {
		t.Errorf("Expected second part text 'And here's my answer', got %q", msg.Content.Parts[1].Text)
	}

	// Check that reasoning details are properly formatted with unknown format
	// Unknown format processes reasoning details like GoogleGeminiV1/OpenAIResponsesV1
	if len(msg.ReasoningDetails) != 4 {
		t.Fatalf("Expected 4 reasoning details (2 reasoning.text + 2 encrypted), got %d", len(msg.ReasoningDetails))
	}

	// First reasoning detail should be reasoning.text type with unknown format
	if msg.ReasoningDetails[0].Type != openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText {
		t.Errorf("Expected first reasoning detail to be reasoning.text type, got %s", msg.ReasoningDetails[0].Type)
	}
	if msg.ReasoningDetails[0].Format != openrouter.ChatCompletionMessageReasoningDetailFormatUnknown {
		t.Errorf("Expected format unknown, got %s", msg.ReasoningDetails[0].Format)
	}
	if msg.ReasoningDetails[0].Text != "First thought" {
		t.Errorf("Expected text 'First thought', got %q", msg.ReasoningDetails[0].Text)
	}

	// Second reasoning detail should be encrypted type with signature split
	if msg.ReasoningDetails[1].Type != openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted {
		t.Errorf("Expected second reasoning detail to be encrypted type, got %s", msg.ReasoningDetails[1].Type)
	}
	if msg.ReasoningDetails[1].Format != openrouter.ChatCompletionMessageReasoningDetailFormatUnknown {
		t.Errorf("Expected format unknown, got %s", msg.ReasoningDetails[1].Format)
	}
	if msg.ReasoningDetails[1].ID != "sig123" {
		t.Errorf("Expected ID 'sig123', got %q", msg.ReasoningDetails[1].ID)
	}
	if msg.ReasoningDetails[1].Data != "data456" {
		t.Errorf("Expected Data 'data456', got %q", msg.ReasoningDetails[1].Data)
	}

	// Third reasoning detail should be reasoning.text for second thought
	if msg.ReasoningDetails[2].Type != openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText {
		t.Errorf("Expected third reasoning detail to be reasoning.text type, got %s", msg.ReasoningDetails[2].Type)
	}
	if msg.ReasoningDetails[2].Format != openrouter.ChatCompletionMessageReasoningDetailFormatUnknown {
		t.Errorf("Expected format unknown, got %s", msg.ReasoningDetails[2].Format)
	}
	if msg.ReasoningDetails[2].Text != "Second thought" {
		t.Errorf("Expected text 'Second thought', got %q", msg.ReasoningDetails[2].Text)
	}

	// Fourth reasoning detail should be encrypted for second signature
	if msg.ReasoningDetails[3].Type != openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted {
		t.Errorf("Expected fourth reasoning detail to be encrypted type, got %s", msg.ReasoningDetails[3].Type)
	}
	if msg.ReasoningDetails[3].Format != openrouter.ChatCompletionMessageReasoningDetailFormatUnknown {
		t.Errorf("Expected format unknown, got %s", msg.ReasoningDetails[3].Format)
	}
	if msg.ReasoningDetails[3].ID != "sig789" {
		t.Errorf("Expected ID 'sig789', got %q", msg.ReasoningDetails[3].ID)
	}
	if msg.ReasoningDetails[3].Data != "data012" {
		t.Errorf("Expected Data 'data012', got %q", msg.ReasoningDetails[3].Data)
	}

	// Check indices are set correctly
	if msg.ReasoningDetails[0].Index != 0 {
		t.Errorf("Expected first reasoning detail to have index 0, got %d", msg.ReasoningDetails[0].Index)
	}
	if msg.ReasoningDetails[1].Index != 0 {
		t.Errorf("Expected second reasoning detail to have index 0 (same as reasoning.text), got %d", msg.ReasoningDetails[1].Index)
	}
	if msg.ReasoningDetails[2].Index != 1 {
		t.Errorf("Expected third reasoning detail to have index 1, got %d", msg.ReasoningDetails[2].Index)
	}
	if msg.ReasoningDetails[3].Index != 1 {
		t.Errorf("Expected fourth reasoning detail to have index 1 (same as reasoning.text), got %d", msg.ReasoningDetails[3].Index)
	}
}

func TestCanonicalOpenRouterMessages_UnknownFormat_WithSignatureWithoutDelimiter(t *testing.T) {
	prof := testProfileWithOptions(func(p *profile.Profile) {
		p.Options.Reasoning.Format = "unknown"
		p.Options.Reasoning.Delimiter = "/"
	})

	sharedMsg := &anthropic.Message{Role: anthropic.MessageRoleAssistant}

	src := []*openrouterChatCompletionMessageWrapper{
		{
			ChatCompletionMessage: &openrouter.ChatCompletionMessage{
				Role: openrouter.ChatCompletionMessageRoleAssistant,
				Content: &openrouter.ChatCompletionMessageContent{
					Type: openrouter.ChatCompletionMessageContentTypeText,
					Text: "Response text",
				},
				ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
					{
						Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
						Text:      "Thinking",
						Signature: "signature_without_delimiter",
					},
				},
			},
			underlyingAnthropicMessage: sharedMsg,
		},
	}

	messages := canonicalOpenRouterMessages(prof, "some-model", src)
	if len(messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(messages))
	}

	msg := messages[0]
	if len(msg.ReasoningDetails) != 2 {
		t.Fatalf("Expected 2 reasoning details, got %d", len(msg.ReasoningDetails))
	}

	// When signature has no delimiter, ID should be empty and Data should contain the whole signature
	encrypted := msg.ReasoningDetails[1]
	if encrypted.Type != openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted {
		t.Errorf("Expected encrypted type, got %s", encrypted.Type)
	}
	if encrypted.ID != "" {
		t.Errorf("Expected empty ID when no delimiter, got %q", encrypted.ID)
	}
	if encrypted.Data != "signature_without_delimiter" {
		t.Errorf("Expected Data to contain whole signature, got %q", encrypted.Data)
	}
}

// testOpenAIProfile creates a test profile for OpenAI provider
func testOpenAIProfile() *profile.Profile {
	return &profile.Profile{
		Name:     "test-openai",
		Provider: "openai",
		Options: &profile.OptionsConfig{
			Strict:                    false,
			ContextWindowResizeFactor: 1.0,
		},
		OpenAI: &profile.OpenAIConfig{
			BaseURL: "https://api.openai.com",
		},
	}
}

// testOpenAICtx creates a context with an OpenAI test profile
func testOpenAICtx() context.Context {
	return profile.WithProfile(context.Background(), testOpenAIProfile())
}

func TestConvertAnthropicRequestToOpenAIRequest_SystemTextOnly(t *testing.T) {
	// When system messages contain only text, they should be converted to instructions
	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-4o",
		MaxTokens: 500,
		System: anthropic.MessageContents{
			{Type: anthropic.MessageContentTypeText, Text: "You are a helpful assistant."},
			{Type: anthropic.MessageContentTypeText, Text: "Always be polite."},
		},
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{Type: anthropic.MessageContentTypeText, Text: "Hello"},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenAIRequest(testOpenAICtx(), src)

	// System should be converted to Instructions (text-only)
	expectedInstructions := "You are a helpful assistant.\n\nAlways be polite."
	if got.Instructions != expectedInstructions {
		t.Errorf("Expected Instructions %q, got %q", expectedInstructions, got.Instructions)
	}

	// Input should only have user message, no system message
	if len(got.Input) != 1 {
		t.Errorf("Expected 1 input item (user message only), got %d", len(got.Input))
	}

	// Verify the input is a user message, not system message
	if got.Input[0].Message == nil {
		t.Fatal("Expected message in input")
	}
	if got.Input[0].Message.Role != openai.ResponseMessageRoleUser {
		t.Errorf("Expected user role, got %s", got.Input[0].Message.Role)
	}
}

func TestConvertAnthropicRequestToOpenAIRequest_SystemWithImage(t *testing.T) {
	// When system messages contain images, they should be converted to input with system role
	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-4o",
		MaxTokens: 500,
		System: anthropic.MessageContents{
			{Type: anthropic.MessageContentTypeText, Text: "You are a helpful assistant."},
			{
				Type: anthropic.MessageContentTypeImage,
				Source: &anthropic.MessageContentSource{
					Type:      "base64",
					MediaType: "image/png",
					Data:      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
				},
			},
		},
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{Type: anthropic.MessageContentTypeText, Text: "Hello"},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenAIRequest(testOpenAICtx(), src)

	// Instructions should be empty when system contains non-text content
	if got.Instructions != "" {
		t.Errorf("Expected empty Instructions, got %q", got.Instructions)
	}

	// Input should have 2 items: system message, then user message
	if len(got.Input) != 2 {
		t.Fatalf("Expected 2 input items, got %d", len(got.Input))
	}

	// First input should be system message
	systemInput := got.Input[0]
	if systemInput.Message == nil {
		t.Fatal("Expected system message in first input")
	}
	if systemInput.Message.Role != openai.ResponseMessageRoleSystem {
		t.Errorf("Expected system role for first input, got %s", systemInput.Message.Role)
	}

	// Verify system message content includes both text and image
	if len(systemInput.Message.Content) != 2 {
		t.Fatalf("Expected 2 content items in system message, got %d", len(systemInput.Message.Content))
	}

	// First content should be text
	if systemInput.Message.Content[0].Text == nil {
		t.Error("Expected text content as first item")
	} else if systemInput.Message.Content[0].Text.Text != "You are a helpful assistant." {
		t.Errorf("Expected text 'You are a helpful assistant.', got %q", systemInput.Message.Content[0].Text.Text)
	}

	// Second content should be image
	if systemInput.Message.Content[1].Image == nil {
		t.Error("Expected image content as second item")
	} else if systemInput.Message.Content[1].Image.ImageUrl == "" {
		t.Error("Expected non-empty image URL")
	}

	// Second input should be user message
	userInput := got.Input[1]
	if userInput.Message == nil {
		t.Fatal("Expected user message in second input")
	}
	if userInput.Message.Role != openai.ResponseMessageRoleUser {
		t.Errorf("Expected user role for second input, got %s", userInput.Message.Role)
	}
}

func TestConvertAnthropicRequestToOpenAIRequest_EmptySystem(t *testing.T) {
	// Empty system should result in empty instructions and no system input
	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-4o",
		MaxTokens: 500,
		System:    anthropic.MessageContents{},
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{Type: anthropic.MessageContentTypeText, Text: "Hello"},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenAIRequest(testOpenAICtx(), src)

	// Instructions should be empty
	if got.Instructions != "" {
		t.Errorf("Expected empty Instructions, got %q", got.Instructions)
	}

	// Input should only have user message
	if len(got.Input) != 1 {
		t.Errorf("Expected 1 input item (user message only), got %d", len(got.Input))
	}
}

func TestConvertAnthropicRequestToOpenAIRequest_NilSystem(t *testing.T) {
	// Nil system should result in empty instructions and no system input
	src := &anthropic.GenerateMessageRequest{
		Model:     "gpt-4o",
		MaxTokens: 500,
		System:    nil,
		Messages: []*anthropic.Message{
			{
				Role: anthropic.MessageRoleUser,
				Content: anthropic.MessageContents{
					{Type: anthropic.MessageContentTypeText, Text: "Hello"},
				},
			},
		},
	}

	got := ConvertAnthropicRequestToOpenAIRequest(testOpenAICtx(), src)

	// Instructions should be empty
	if got.Instructions != "" {
		t.Errorf("Expected empty Instructions, got %q", got.Instructions)
	}

	// Input should only have user message
	if len(got.Input) != 1 {
		t.Errorf("Expected 1 input item (user message only), got %d", len(got.Input))
	}
}
