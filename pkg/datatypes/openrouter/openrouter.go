package openrouter

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	"github.com/samber/lo"
	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/utils"
)

func init() {
	viper.SetDefault("openrouter.allowed_providers", []string{ProviderAnthropic})
}

func WithIdentity(referer string, title string) func(*http.Request) {
	return func(req *http.Request) {
		req.Header.Set("HTTP-Referer", referer)
		req.Header.Set("X-Title", title)
	}
}

func WithProviderPreference(pref *ProviderPreference) func(*http.Request) {
	return func(req *http.Request) {
		if req.GetBody != nil {
			if r, err := req.GetBody(); err == nil {
				var data *CreateChatCompletionRequest
				if err = json.NewDecoder(r).Decode(&data); err == nil {
					data.Provider = pref
					if newBody, err := json.Marshal(data); err == nil {
						if oldBody := req.Body; oldBody != nil {
							oldBody.Close()
						}
						req.ContentLength = int64(len(newBody))
						req.Body = io.NopCloser(bytes.NewReader(newBody))
						req.GetBody = func() (io.ReadCloser, error) {
							return io.NopCloser(bytes.NewReader(newBody)), nil
						}
					}
				}
			}
		}
	}
}

type Error struct {
	Inner struct {
		Code     int            `json:"code"`
		Message  string         `json:"message"`
		Metadata map[string]any `json:"metadata"`
	} `json:"error"`

	statusCode int
}

func (e *Error) Error() string {
	return fmt.Sprintf("(%d) %s", e.Inner.Code, e.Inner.Message)
}

func (e *Error) Type() string {
	switch e.Inner.Code / 100 {
	case 4:
		if e.Inner.Code == 429 {
			return "overloaded_error"
		}
		return "invalid_request_error"
	case 5:
		return "internal_server_error"
	default:
		return "unknown_error"
	}
}

func (e *Error) Message() string              { return e.Inner.Message }
func (e *Error) Source() string               { return "openrouter" }
func (e *Error) StatusCode() int              { return e.statusCode }
func (e *Error) SetStatusCode(statusCode int) { e.statusCode = statusCode }

type ChatCompletionStream iter.Seq2[*ChatCompletionChunk, error]

// CreateChatCompletionRequest follows OpenRouter request format
// reference: https://openrouter.ai/docs/api-reference/
type CreateChatCompletionRequest struct {
	Messages            []*ChatCompletionMessage       `json:"messages"`
	Model               string                         `json:"model"`
	MaxCompletionTokens *int                           `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int                           `json:"max_tokens,omitempty"`
	ParallelToolCalls   *bool                          `json:"parallel_tool_calls,omitempty"`
	ReasoningEffort     *ChatCompletionReasoningEffort `json:"reasoning_effort,omitempty"`
	Reasoning           *ChatCompletionReasoning       `json:"reasoning,omitempty"`
	ResponseFormat      *ChatCompletionResponseFormat  `json:"response_format,omitempty"`
	Stop                ChatCompletionStop             `json:"stop,omitempty"`
	StreamOptions       *ChatCompletionStreamOptions   `json:"stream_options,omitempty"`
	Temperature         *float64                       `json:"temperature,omitempty"`
	ToolChoice          *ChatCompletionToolChoice      `json:"tool_choice,omitempty"`
	Tools               []*ChatCompletionTool          `json:"tools,omitempty"`
	TopP                *float64                       `json:"top_p,omitempty"`
	TopK                *int                           `json:"top_k,omitempty"`
	User                string                         `json:"user,omitempty"`
	Stream              utils.True                     `json:"stream"`
	Provider            *ProviderPreference            `json:"provider,omitempty"`
}

const (
	ProviderGoogleVertex       = "google-vertex"
	ProviderGoogleVertexGlobal = "google-vertex/global"
	ProviderGoogleVertexEurope = "google-vertex/europe"
	ProviderAnthropic          = "anthropic"
	ProviderAmazonBedrock      = "amazon-bedrock"
)

type ProviderPreference struct {
	Order             []string                      `json:"order,omitempty"`
	AllowFallbacks    *bool                         `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool                         `json:"require_parameters,omitempty"`
	DataCollection    *ProviderDataCollectionPolicy `json:"data_collection,omitempty"`
	Only              []string                      `json:"only,omitempty"`
	Ignore            []string                      `json:"ignore,omitempty"`
	Quantizations     []ProviderQuantizationLevel   `json:"quantizations,omitempty"`
	Sort              *ProviderSortMethod           `json:"sort,omitempty"`
	MaxPrice          *ProviderMaxPrice             `json:"max_price,omitempty"`
	Experimental      *ProviderExperimental         `json:"experimental,omitempty"`
}

type ProviderDataCollectionPolicy string

const (
	ProviderDataCollectionPolicyAllow ProviderDataCollectionPolicy = "allow"
	ProviderDataCollectionPolicyDeny  ProviderDataCollectionPolicy = "deny"
)

type ProviderQuantizationLevel string

const (
	ProviderQuantizationLevelInt4    ProviderQuantizationLevel = "int4"
	ProviderQuantizationLevelInt8    ProviderQuantizationLevel = "int8"
	ProviderQuantizationLevelFP4     ProviderQuantizationLevel = "fp4"
	ProviderQuantizationLevelFP6     ProviderQuantizationLevel = "fp6"
	ProviderQuantizationLevelFP8     ProviderQuantizationLevel = "fp8"
	ProviderQuantizationLevelFP16    ProviderQuantizationLevel = "fp16"
	ProviderQuantizationLevelBF16    ProviderQuantizationLevel = "bf16"
	ProviderQuantizationLevelFP32    ProviderQuantizationLevel = "fp32"
	ProviderQuantizationLevelUnknown ProviderQuantizationLevel = "unknown"
)

type ProviderSortMethod string

const (
	ProviderSortMethodPrice      ProviderSortMethod = "price"
	ProviderSortMethodThroughput ProviderSortMethod = "throughput"
	ProviderSortMethodLatency    ProviderSortMethod = "latency"
)

type ProviderMaxPrice struct {
	Prompt     ProviderMaxPriceValue `json:"prompt,omitempty"`
	Completion ProviderMaxPriceValue `json:"completion,omitempty"`
	Image      ProviderMaxPriceValue `json:"image,omitempty"`
	Audio      ProviderMaxPriceValue `json:"audio,omitempty"`
	Request    ProviderMaxPriceValue `json:"request,omitempty"`
}

type ProviderMaxPriceValue interface {
	maxPrice()
}

type (
	ProviderMaxPriceNumber float64
	ProviderMaxPriceString string
)

func (ProviderMaxPriceNumber) maxPrice() {}
func (ProviderMaxPriceString) maxPrice() {}

type ProviderExperimental struct {
}

type ChatCompletionReasoning struct {
	Effort    ChatCompletionReasoningEffort `json:"effort,omitempty"`
	MaxTokens int                           `json:"max_tokens,omitempty"`
	Exclude   bool                          `json:"exclude,omitempty"`
	Enabled   bool                          `json:"enabled,omitempty"`
}

type ChatCompletionReasoningEffort string

const (
	ChatCompletionReasoningEffortLow    ChatCompletionReasoningEffort = "low"
	ChatCompletionReasoningEffortMedium ChatCompletionReasoningEffort = "medium"
	ChatCompletionReasoningEffortHigh   ChatCompletionReasoningEffort = "high"
)

type ChatCompletionStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ChatCompletionTool struct {
	Type     ChatCompletionMessageToolCallType `json:"type"`
	Function *ChatCompletionFunction           `json:"function"`
}

type ChatCompletionFunction struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description,omitempty"`
	Strict      bool                           `json:"strict,omitempty"`
	Parameters  ChatCompletionJSONSchemaObject `json:"parameters,omitempty"`
}

type ChatCompletionJSONSchemaObject []byte

func (param ChatCompletionJSONSchemaObject) MarshalJSON() ([]byte, error) {
	return []byte(param), nil
}

func (param *ChatCompletionJSONSchemaObject) UnmarshalJSON(b []byte) error {
	*param = append((*param)[:0], b...)
	return nil
}

type ChatCompletionMessage struct {
	Role             ChatCompletionRole                      `json:"role"`
	Content          *ChatCompletionMessageContent           `json:"content"`
	ToolCallID       string                                  `json:"tool_call_id,omitempty"`
	Refusal          *string                                 `json:"refusal,omitempty"`
	ToolCalls        []*ChatCompletionToolCall               `json:"tool_calls,omitempty"`
	Reasoning        string                                  `json:"reasoning,omitempty"`
	ReasoningDetails []*ChatCompletionMessageReasoningDetail `json:"reasoning_details,omitempty"`
}

type ChatCompletionRole string

const (
	ChatCompletionMessageRoleDeveloper ChatCompletionRole = "developer"
	ChatCompletionMessageRoleSystem    ChatCompletionRole = "system"
	ChatCompletionMessageRoleUser      ChatCompletionRole = "user"
	ChatCompletionMessageRoleAssistant ChatCompletionRole = "assistant"
	ChatCompletionMessageRoleTool      ChatCompletionRole = "tool"
)

type ChatCompletionMessageContent struct {
	Type  ChatCompletionMessageContentType
	Text  string
	Parts []*ChatCompletionMessageContentPart
}

func (c ChatCompletionMessageContent) IsText() bool {
	return c.Type == ChatCompletionMessageContentTypeText
}

func (c ChatCompletionMessageContent) IsParts() bool {
	return c.Type == ChatCompletionMessageContentTypeParts
}

func (c ChatCompletionMessageContent) MarshalJSON() ([]byte, error) {
	switch c.Type {
	case ChatCompletionMessageContentTypeText:
		return json.Marshal(c.Text)
	case ChatCompletionMessageContentTypeParts:
		return json.Marshal(c.Parts)
	}
	return json.Marshal(c.Text)
}

func (c *ChatCompletionMessageContent) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '[':
			c.Type = ChatCompletionMessageContentTypeParts
			return json.Unmarshal(data, &c.Parts)
		case '"':
			c.Type = ChatCompletionMessageContentTypeText
			return json.Unmarshal(data, &c.Text)
		default:
			return errors.New("message content should be a string or an array")
		}
	}
	return errors.New("empty message content")
}

type ChatCompletionMessageContentType string

const (
	ChatCompletionMessageContentTypeText  ChatCompletionMessageContentType = "text"
	ChatCompletionMessageContentTypeParts ChatCompletionMessageContentType = "parts"
)

type ChatCompletionMessageContentPart struct {
	Type         ChatCompletionMessageContentPartType      `json:"type"`
	Text         string                                    `json:"text,omitempty"`
	Refusal      *string                                   `json:"refusal,omitempty"`
	ImageUrl     *ChatCompletionMessageContentPartImageUrl `json:"image_url,omitempty"`
	CacheControl *ChatCompletionMessageCacheControl        `json:"cache_control,omitempty"`
}

func (part ChatCompletionMessageContentPart) IsText() bool {
	return part.Type == ChatCompletionMessageContentPartTypeText
}

func (part ChatCompletionMessageContentPart) IsRefusal() bool {
	return part.Type == ChatCompletionMessageContentPartTypeRefusal
}

func (part ChatCompletionMessageContentPart) IsImage() bool {
	return part.Type == ChatCompletionMessageContentPartTypeImage
}

type ChatCompletionMessageContentPartType string

const (
	ChatCompletionMessageContentPartTypeText    ChatCompletionMessageContentPartType = "text"
	ChatCompletionMessageContentPartTypeRefusal ChatCompletionMessageContentPartType = "refusal"
	ChatCompletionMessageContentPartTypeImage   ChatCompletionMessageContentPartType = "image_url"
)

type ChatCompletionMessageCacheControl struct {
	Type ChatCompletionMessageCacheControlType `json:"type"`
	TTL  ChatCompletionMessageCacheControlTTL  `json:"ttl,omitempty"`
}

type (
	ChatCompletionMessageCacheControlType string
	ChatCompletionMessageCacheControlTTL  string
)

type ChatCompletionMessageContentPartImageUrl struct {
	Url    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ChatCompletionToolCall struct {
	Index    int                                    `json:"index"`
	ID       string                                 `json:"id"`
	Type     ChatCompletionMessageToolCallType      `json:"type"`
	Function *ChatCompletionMessageToolCallFunction `json:"function"`
}

type ChatCompletionMessageToolCallType string

const (
	ChatCompletionMessageToolCallTypeFunction ChatCompletionMessageToolCallType = "function"
)

type ChatCompletionMessageReasoningDetail struct {
	Type      ChatCompletionMessageReasoningDetailType   `json:"type"`
	Text      string                                     `json:"text"`
	Summary   string                                     `json:"summary"`
	ID        string                                     `json:"id"`
	Data      string                                     `json:"data"`
	Signature string                                     `json:"signature"`
	Format    ChatCompletionMessageReasoningDetailFormat `json:"format"`
	Index     int                                        `json:"index"`
}

type ChatCompletionMessageReasoningDetailType string

const (
	ChatCompletionMessageReasoningDetailTypeReasoningText ChatCompletionMessageReasoningDetailType = "reasoning.text"
	ChatCompletionMessageReasoningDetailTypeSummary       ChatCompletionMessageReasoningDetailType = "reasoning.summary"
	ChatCompletionMessageReasoningDetailTypeEncrypted     ChatCompletionMessageReasoningDetailType = "reasoning.encrypted"
)

type ChatCompletionMessageReasoningDetailFormat string

const (
	ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1 ChatCompletionMessageReasoningDetailFormat = "anthropic-claude-v1"
	ChatCompletionMessageReasoningDetailFormatOpenAIResponsesV1 ChatCompletionMessageReasoningDetailFormat = "openai-responses-v1"
)

type ChatCompletionMessageToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatCompletionResponseFormat struct {
	Type       ChatCompletionResponseFormatType        `json:"type"`
	JSONSchema *ChatCompletionResponseFormatJSONSchema `json:"json_schema,omitempty"`
}

type ChatCompletionResponseFormatType string

const (
	ChatCompletionResponseFormatTypeText       ChatCompletionResponseFormatType = "text"
	ChatCompletionResponseFormatTypeJSONObject ChatCompletionResponseFormatType = "json_object"
	ChatCompletionResponseFormatTypeJSONSchema ChatCompletionResponseFormatType = "json_schema"
)

func (format ChatCompletionResponseFormat) MarshalJSON() ([]byte, error) {
	switch format.Type {
	case ChatCompletionResponseFormatTypeText:
		return json.Marshal(ChatCompletionResponseFormatTypeText)
	default:
		return json.Marshal(struct {
			Type       ChatCompletionResponseFormatType        `json:"type"`
			JSONSchema *ChatCompletionResponseFormatJSONSchema `json:"json_schema,omitempty"`
		}{
			Type:       format.Type,
			JSONSchema: format.JSONSchema,
		})
	}
}

func (format *ChatCompletionResponseFormat) UnmarshalJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	tok, err := decoder.Token()
	if err != nil {
		return err
	}
	if tok, ok := tok.(string); ok {
		if tok == string(ChatCompletionResponseFormatTypeText) {
			format.Type = ChatCompletionResponseFormatTypeText
			return nil
		} else {
			return fmt.Errorf("invalid response_format: expects string value \"text\", got %s", tok)
		}
	} else {
		var ir struct {
			Type       ChatCompletionResponseFormatType        `json:"type"`
			JSONSchema *ChatCompletionResponseFormatJSONSchema `json:"json_schema,omitempty"`
		}
		if err := json.Unmarshal(data, &ir); err != nil {
			return err
		}
		switch ir.Type {
		case ChatCompletionResponseFormatTypeJSONObject, ChatCompletionResponseFormatTypeJSONSchema:
		default:
			return errors.New("invalid response_format, available types are 'text', 'json_object', 'json_schema'")
		}
		format.Type = ir.Type
		format.JSONSchema = ir.JSONSchema
		return nil
	}
}

type ChatCompletionResponseFormatJSONSchema struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description,omitempty"`
	Strict      bool                           `json:"strict,omitempty"`
	Schema      ChatCompletionJSONSchemaObject `json:"schema,omitempty"`
}

type ChatCompletionChunk struct {
	ID                string                       `json:"id"`
	Provider          string                       `json:"provider"`
	Model             string                       `json:"model"`
	Created           int64                        `json:"created"`
	Object            string                       `json:"object"`
	Choices           []*ChatCompletionChunkChoice `json:"choices"`
	ServiceTier       string                       `json:"service_tier"`
	SystemFingerprint string                       `json:"system_fingerprint"`
	Usage             *ChatCompletionUsage         `json:"usage"`
}

type ChatCompletionChunkChoice struct {
	Index              int                             `json:"index"`
	Delta              *ChatCompletionChunkChoiceDelta `json:"delta,omitempty"`
	Logprobs           *ChatCompletionLogprobs         `json:"logprobs,omitempty"`
	FinishReason       ChatCompletionFinishReason      `json:"finish_reason,omitempty"`
	NativeFinishReason string                          `json:"native_finish_reason,omitempty"`
}

type ChatCompletionChunkChoiceDelta struct {
	Role             ChatCompletionRole                      `json:"role,omitempty"`
	Content          string                                  `json:"content,omitempty"`
	Refusal          *string                                 `json:"refusal,omitempty"`
	ToolCalls        []*ChatCompletionToolCall               `json:"tool_calls,omitempty"`
	Reasoning        string                                  `json:"reasoning,omitempty"`
	ReasoningDetails []*ChatCompletionMessageReasoningDetail `json:"reasoning_details,omitempty"`
}

type ChatCompletionLogprobs struct {
	Content []*ChatCompletionLogprob `json:"content"`
	Refusal []*ChatCompletionLogprob `json:"refusal"`
}

type ChatCompletionLogprob struct {
	Token       string                   `json:"token"`
	Logprob     float64                  `json:"logprob"`
	Bytes       []byte                   `json:"bytes"`
	TopLogprobs []*ChatCompletionLogprob `json:"top_logprobs"`
}

type ChatCompletionUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type ChatCompletion struct {
	ID       string                  `json:"id"`
	Provider string                  `json:"provider"`
	Model    string                  `json:"model"`
	Created  int64                   `json:"created"`
	Object   string                  `json:"object"`
	Choices  []*ChatCompletionChoice `json:"choices"`
	Usage    *ChatCompletionUsage    `json:"usage"`
}

func (c ChatCompletion) GetPromptTokens() int64 {
	if c.Usage == nil {
		return 0
	}
	return c.Usage.PromptTokens
}

func (c ChatCompletion) GetCompletionTokens() int64 {
	if c.Usage == nil {
		return 0
	}
	return c.Usage.CompletionTokens
}

type ChatCompletionChoice struct {
	Index              int                        `json:"index"`
	Message            *ChatCompletionMessage     `json:"message,omitempty"`
	Logprobs           *ChatCompletionLogprobs    `json:"logprobs,omitempty"`
	FinishReason       ChatCompletionFinishReason `json:"finish_reason,omitempty"`
	NativeFinishReason string                     `json:"native_finish_reason,omitempty"`
}

type ChatCompletionFinishReason string

const (
	ChatCompletionFinishReasonStop          ChatCompletionFinishReason = "stop"
	ChatCompletionFinishReasonLength        ChatCompletionFinishReason = "length"
	ChatCompletionFinishReasonContentFilter ChatCompletionFinishReason = "content_filter"
	ChatCompletionFinishReasonToolCalls     ChatCompletionFinishReason = "tool_calls"
)

func NewChatCompletionBuilder() *ChatCompletionBuilder {
	return &ChatCompletionBuilder{
		Choices: make([]*ChatCompletionChoiceBuilder, 0, 1),
	}
}

type ChatCompletionBuilder struct {
	ID       string
	Provider string
	Model    string
	Created  int64
	Object   string
	Choices  []*ChatCompletionChoiceBuilder
	Usage    *ChatCompletionUsage
}

func (builder *ChatCompletionBuilder) Build() *ChatCompletion {
	c := &ChatCompletion{
		ID:       builder.ID,
		Provider: builder.Provider,
		Model:    builder.Model,
		Created:  builder.Created,
		Object:   builder.Object,
		Choices:  make([]*ChatCompletionChoice, len(builder.Choices)),
		Usage:    builder.Usage,
	}
	for i, choice := range builder.Choices {
		c.Choices[i] = choice.Build()
	}
	if c.Usage != nil && c.Usage.TotalTokens == 0 {
		c.Usage.TotalTokens = c.Usage.PromptTokens + c.Usage.CompletionTokens
	}
	return c
}

func (builder *ChatCompletionBuilder) Add(chunk *ChatCompletionChunk) {
	if chunk == nil {
		return
	}
	if builder.ID == "" {
		builder.ID = chunk.ID
	}
	if builder.Provider == "" {
		builder.Provider = chunk.Provider
	}
	if builder.Model == "" {
		builder.Model = chunk.Model
	}
	if builder.Created == 0 {
		builder.Created = chunk.Created
	}
	if builder.Object == "" {
		builder.Object = chunk.Object
	}
	for _, choice := range chunk.Choices {
		if choice.Index >= len(builder.Choices) {
			for range (choice.Index - len(builder.Choices)) + 1 {
				builder.Choices = append(builder.Choices, &ChatCompletionChoiceBuilder{Index: -1})
			}
		}
		choiceBuilder := builder.Choices[choice.Index]
		choiceBuilder.Add(choice)
	}
	if chunk.Usage != nil {
		if builder.Usage == nil {
			builder.Usage = &ChatCompletionUsage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
		} else {
			if chunk.Usage.PromptTokens > 0 {
				builder.Usage.PromptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				builder.Usage.CompletionTokens = chunk.Usage.CompletionTokens
			}
			if chunk.Usage.TotalTokens > 0 {
				builder.Usage.TotalTokens = chunk.Usage.TotalTokens
			}
		}
	}
}

type ChatCompletionChoiceBuilder struct {
	Index              int
	Message            *ChatCompletionMessageBuilder
	Logprobs           *ChatCompletionLogprobs
	FinishReason       ChatCompletionFinishReason
	NativeFinishReason string
}

func (builder *ChatCompletionChoiceBuilder) Build() *ChatCompletionChoice {
	c := &ChatCompletionChoice{
		Index:              builder.Index,
		Message:            builder.Message.Build(),
		Logprobs:           builder.Logprobs,
		FinishReason:       builder.FinishReason,
		NativeFinishReason: builder.NativeFinishReason,
	}
	return c
}

func (builder *ChatCompletionChoiceBuilder) Add(choice *ChatCompletionChunkChoice) {
	if choice == nil {
		return
	}
	if builder.Index == -1 {
		builder.Index = choice.Index
	}
	if builder.Index != choice.Index {
		return
	}
	if builder.Message == nil {
		builder.Message = &ChatCompletionMessageBuilder{}
	}
	builder.Message.Add(choice.Delta)
	if choice.Logprobs != nil {
		if builder.Logprobs == nil {
			builder.Logprobs = &ChatCompletionLogprobs{}
		}
		builder.Logprobs.Content = append(builder.Logprobs.Content, choice.Logprobs.Content...)
		builder.Logprobs.Refusal = append(builder.Logprobs.Refusal, choice.Logprobs.Refusal...)
	}
	if builder.FinishReason == "" {
		builder.FinishReason = choice.FinishReason
	}
	if builder.NativeFinishReason == "" {
		builder.NativeFinishReason = choice.NativeFinishReason
	}
}

type ChatCompletionMessageBuilder struct {
	Role             ChatCompletionRole
	Content          []byte
	Refusal          []byte
	ToolCalls        []*ChatCompletionMessageToolCallBuilder
	Reasoning        []byte
	ReasoningDetails []*ChatCompletionMessageReasoningDetailBuilder
}

func (builder *ChatCompletionMessageBuilder) Build() *ChatCompletionMessage {
	var refusal *string
	if builder.Refusal == nil {
		refusal = nil
	} else {
		refusal = lo.ToPtr(string(builder.Refusal))
	}
	c := &ChatCompletionMessage{
		Role:             builder.Role,
		Content:          &ChatCompletionMessageContent{Type: ChatCompletionMessageContentTypeText, Text: string(builder.Content)},
		Refusal:          refusal,
		ToolCalls:        make([]*ChatCompletionToolCall, len(builder.ToolCalls)),
		Reasoning:        string(builder.Reasoning),
		ReasoningDetails: make([]*ChatCompletionMessageReasoningDetail, len(builder.ReasoningDetails)),
	}
	for i, toolCall := range builder.ToolCalls {
		c.ToolCalls[i] = toolCall.Build()
	}
	for i, reasoningDetail := range builder.ReasoningDetails {
		c.ReasoningDetails[i] = reasoningDetail.Build()
	}
	return c
}

func (builder *ChatCompletionMessageBuilder) Add(delta *ChatCompletionChunkChoiceDelta) {
	if delta == nil {
		return
	}
	if builder.Role == "" {
		builder.Role = delta.Role
	}
	builder.Content = append(builder.Content, delta.Content...)
	if delta.Refusal != nil {
		builder.Refusal = append(builder.Refusal, *delta.Refusal...)
	}
	for _, toolCall := range delta.ToolCalls {
		if toolCall.Index >= len(builder.ToolCalls) {
			for range (toolCall.Index - len(builder.ToolCalls)) + 1 {
				builder.ToolCalls = append(builder.ToolCalls, &ChatCompletionMessageToolCallBuilder{})
			}
		}
		toolCallBuilder := builder.ToolCalls[toolCall.Index]
		toolCallBuilder.Add(toolCall)
	}
	for _, reasoningDetail := range delta.ReasoningDetails {
		if reasoningDetail.Index >= len(builder.ReasoningDetails) {
			for range (reasoningDetail.Index - len(builder.ReasoningDetails)) + 1 {
				builder.ReasoningDetails = append(builder.ReasoningDetails, &ChatCompletionMessageReasoningDetailBuilder{})
			}
		}
		reasoningDetailBuilder := builder.ReasoningDetails[reasoningDetail.Index]
		reasoningDetailBuilder.Add(reasoningDetail)
	}
	builder.Reasoning = append(builder.Reasoning, delta.Reasoning...)
}

type ChatCompletionMessageToolCallBuilder struct {
	ID       string
	Type     ChatCompletionMessageToolCallType
	Function *ChatCompletionMessageToolCallFunctionBuilder
}

func (builder *ChatCompletionMessageToolCallBuilder) Build() *ChatCompletionToolCall {
	c := &ChatCompletionToolCall{
		ID:       builder.ID,
		Type:     builder.Type,
		Function: builder.Function.Build(),
	}
	return c
}

func (builder *ChatCompletionMessageToolCallBuilder) Add(toolCall *ChatCompletionToolCall) {
	if toolCall == nil {
		return
	}
	if builder.ID == "" {
		builder.ID = toolCall.ID
	}
	if builder.Type == "" {
		builder.Type = toolCall.Type
	}
	if builder.Function == nil {
		builder.Function = &ChatCompletionMessageToolCallFunctionBuilder{}
	}
	builder.Function.Add(toolCall.Function)
}

type ChatCompletionMessageToolCallFunctionBuilder struct {
	Name      string
	Arguments []byte
}

func (builder *ChatCompletionMessageToolCallFunctionBuilder) Build() *ChatCompletionMessageToolCallFunction {
	c := &ChatCompletionMessageToolCallFunction{
		Name:      builder.Name,
		Arguments: string(builder.Arguments),
	}
	return c
}

func (builder *ChatCompletionMessageToolCallFunctionBuilder) Add(function *ChatCompletionMessageToolCallFunction) {
	if function == nil {
		return
	}
	if builder.Name == "" {
		builder.Name = function.Name
	}
	builder.Arguments = append(builder.Arguments, function.Arguments...)
}

type ChatCompletionMessageReasoningDetailBuilder struct {
	Type      ChatCompletionMessageReasoningDetailType   `json:"type"`
	Text      []byte                                     `json:"text"`
	Signature string                                     `json:"signature"`
	Format    ChatCompletionMessageReasoningDetailFormat `json:"format"`
	Index     int                                        `json:"index"`
}

func (builder *ChatCompletionMessageReasoningDetailBuilder) Build() *ChatCompletionMessageReasoningDetail {
	c := &ChatCompletionMessageReasoningDetail{
		Type:      builder.Type,
		Text:      string(builder.Text),
		Signature: builder.Signature,
		Format:    builder.Format,
		Index:     builder.Index,
	}
	return c
}

func (builder *ChatCompletionMessageReasoningDetailBuilder) Add(reasoningDetail *ChatCompletionMessageReasoningDetail) {
	if reasoningDetail == nil {
		return
	}
	if builder.Type == "" {
		builder.Type = reasoningDetail.Type
	}
	builder.Text = append(builder.Text, reasoningDetail.Text...)
	if builder.Signature == "" {
		builder.Signature = reasoningDetail.Signature
	}
	if builder.Format == "" {
		builder.Format = reasoningDetail.Format
	}
}

type ChatCompletionStop []string

func (stop *ChatCompletionStop) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case 'n':
			return nil
		case '"':
			*stop = make(ChatCompletionStop, 1)
			return json.Unmarshal(data, &(*stop)[0])
		case '[':
			var array []string
			if err := json.Unmarshal(data, &array); err != nil {
				return err
			}
			*stop = ChatCompletionStop(array)
		default:
			return errors.New("stop should be a string, an array or null")
		}
	}
	return errors.New("empty stop")
}

type ChatCompletionToolChoice struct {
	Mode ChatCompletionToolChoiceType
	Tool *ChatCompletionTool
}

type ChatCompletionToolChoiceType string

const (
	ChatCompletionToolChoiceTypeAuto     ChatCompletionToolChoiceType = "auto"
	ChatCompletionToolChoiceTypeNone     ChatCompletionToolChoiceType = "none"
	ChatCompletionToolChoiceTypeRequired ChatCompletionToolChoiceType = "required"
)

func (toolChoice ChatCompletionToolChoice) MarshalJSON() ([]byte, error) {
	if toolChoice.Mode == "" && toolChoice.Tool == nil {
		return json.Marshal(nil)
	}
	if toolChoice.Tool != nil {
		return json.Marshal(toolChoice.Tool)
	}
	return json.Marshal(toolChoice.Mode)
}

func (toolChoice *ChatCompletionToolChoice) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case 'n':
			return nil
		case '"':
			return json.Unmarshal(data, &toolChoice.Mode)
		case '{':
			return json.Unmarshal(data, &toolChoice.Tool)
		default:
			return errors.New("tool_choice should be a string, an object or null")
		}
	}
	return errors.New("empty tool_choice")
}
