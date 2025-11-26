package anthropic

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/x5iu/claude-code-adapter/pkg/utils"
)

func WithBetaFeatures(features ...string) func(*http.Request) {
	return func(req *http.Request) {
		req.Header.Add(HeaderBeta, strings.Join(features, ","))
	}
}

const (
	BetaFeatureInterleavedThinking20250514 = "interleaved-thinking-2025-05-14"
)

const (
	HeaderAPIKey  = "x-api-key"
	HeaderVersion = "anthropic-version"
	HeaderBeta    = "anthropic-beta"
)

const (
	ErrorContentType = "error"
)

const (
	InvalidRequestError = "invalid_request_error"
	AuthenticationError = "authentication_error"
	PermissionError     = "permission_error"
	NotFoundError       = "not_found_error"
	RequestTooLarge     = "request_too_large"
	RateLimitError      = "rate_limit_error"
	APIError            = "api_error"
	OverloadedError     = "overloaded_error"
)

type Error struct {
	ContentType string      `json:"type"`
	Inner       *InnerError `json:"error"`

	statusCode int
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", e.Type(), e.Message())
}

func (e *Error) Type() string                 { return e.Inner.Type }
func (e *Error) Message() string              { return e.Inner.Message }
func (e *Error) Source() string               { return "anthropic" }
func (e *Error) StatusCode() int              { return e.statusCode }
func (e *Error) SetStatusCode(statusCode int) { e.statusCode = statusCode }

type InnerError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type StreamError struct {
	ErrType    string `json:"type"`
	ErrMessage string `json:"message"`
}

func (e *StreamError) Error() string {
	return fmt.Sprintf("%s: %s", e.Type(), e.Message())
}

func (e *StreamError) Type() string    { return e.ErrType }
func (e *StreamError) Message() string { return e.ErrMessage }
func (e *StreamError) Source() string  { return "anthropic" }

// GenerateMessageRequest follows Anthropic request format
// reference: https://docs.anthropic.com/en/api/messages
type GenerateMessageRequest struct {
	System        MessageContents `json:"system,omitempty"`
	Model         string          `json:"model"`
	Messages      []*Message      `json:"messages"`
	MaxTokens     int             `json:"max_tokens"`
	Metadata      *Metadata       `json:"metadata,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Thinking      *Thinking       `json:"thinking,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	Tools         []*Tool         `json:"tools,omitempty"`
	Temperature   float64         `json:"temperature,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stream        utils.True      `json:"stream"`
}

type CountTokensRequest struct {
	System     MessageContents `json:"system,omitempty"`
	Model      string          `json:"model"`
	Messages   []*Message      `json:"messages"`
	Thinking   *Thinking       `json:"thinking,omitempty"`
	ToolChoice *ToolChoice     `json:"tool_choice,omitempty"`
	Tools      []*Tool         `json:"tools,omitempty"`
}

type Message struct {
	ID           string          `json:"id,omitempty"`
	Type         MessageType     `json:"type,omitempty"`
	Role         MessageRole     `json:"role"`
	Content      MessageContents `json:"content"`
	Model        string          `json:"model,omitempty"`
	StopReason   *StopReason     `json:"stop_reason,omitempty"`
	StopSequence *string         `json:"stop_sequence,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
}

type MessageType string

const (
	MessageTypeMessage MessageType = "message"
)

type MessageRole string

const (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
)

type StopReason string

const (
	StopReasonEndTurn      StopReason = "end_turn"
	StopReasonMaxTokens    StopReason = "max_tokens"
	StopReasonStopSequence StopReason = "stop_sequence"
	StopReasonToolUse      StopReason = "tool_use"
	StopReasonPauseTurn    StopReason = "pause_turn"
	StopReasonRefusal      StopReason = "refusal"
)

type MessageContentType string

const (
	MessageContentTypeText                MessageContentType = "text"
	MessageContentTypeImage               MessageContentType = "image"
	MessageContentTypeToolUse             MessageContentType = "tool_use"
	MessageContentTypeToolResult          MessageContentType = "tool_result"
	MessageContentTypeThinking            MessageContentType = "thinking"
	MessageContentTypeRedactedThinking    MessageContentType = "redacted_thinking"
	MessageContentTypeServerToolUse       MessageContentType = "server_tool_use"
	MessageContentTypeWebSearchToolResult MessageContentType = "web_search_tool_result"
	MessageContentTypeWebSearchResult     MessageContentType = "web_search_result"
)

type MessageContents []*MessageContent

func (mc MessageContents) MarshalJSON() ([]byte, error) {
	if mc == nil {
		return []byte("[]"), nil
	}
	return json.Marshal([]*MessageContent(mc))
}

func (mc *MessageContents) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', 'n', '\t':
		case '[':
			var contents []*MessageContent
			if err := json.Unmarshal(data, &contents); err != nil {
				return err
			}
			*mc = contents
			return nil
		case '"':
			var content string
			if err := json.Unmarshal(data, &content); err != nil {
				return err
			}
			*mc = append(*mc, &MessageContent{
				Type: MessageContentTypeText,
				Text: content,
			})
			return nil
		default:
			return errors.New("message content should be a string or an array")
		}
	}
	return errors.New("empty message content")
}

type MessageContent struct {
	Type      MessageContentType    `json:"type"`
	Text      string                `json:"text,omitempty"`
	Source    *MessageContentSource `json:"source,omitempty"`
	Thinking  string                `json:"thinking,omitempty"`
	Signature string                `json:"signature,omitempty"`
	Data      string                `json:"data,omitempty"`
	ID        string                `json:"id,omitempty"`
	Name      string                `json:"name,omitempty"`
	Input     json.RawMessage       `json:"input,omitempty"`
	ToolUseID string                `json:"tool_use_id,omitempty"`
	Content   MessageContents       `json:"content,omitempty"`

	Title            string  `json:"title,omitempty"`
	Url              string  `json:"url,omitempty"`
	EncryptedContent string  `json:"encrypted_content,omitempty"`
	PageAge          *string `json:"page_age,omitempty"`

	// Citation are always enabled for web search
	// reference: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool#citations
	Citations []*Citation `json:"citation,omitempty"`

	// CacheControl enables prompt caching from Anthropic
	// reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type MessageContentSource struct {
	Type      MessageContentType `json:"type"`
	MediaType string             `json:"media_type,omitempty"`
	Data      string             `json:"data,omitempty"`
}

type CacheControl struct {
	Type MessageCacheControlType `json:"type"`
	TTL  MessageCacheControlTTL  `json:"ttl,omitempty"`
}

type MessageCacheControlType string

const (
	MessageCacheControlTypeEphemeral MessageCacheControlType = "ephemeral"
)

type MessageCacheControlTTL string

const (
	MessageCacheControlTTL5Minutes MessageCacheControlTTL = "5m"
	MessageCacheControlTTL1Hour    MessageCacheControlTTL = "1h"
)

type Citation struct {
	Type           CitationType `json:"type"`
	URL            string       `json:"url"`
	Title          string       `json:"title"`
	EncryptedIndex string       `json:"encrypted_index"`
	CitedText      string       `json:"cited_text"`
}

type CitationType string

const (
	CitationTypeWebSearchResultLocation CitationType = "web_search_result_location"
)

type MessageContentDeltaType string

const (
	MessageContentDeltaTypeTextDelta      MessageContentDeltaType = "text_delta"
	MessageContentDeltaTypeInputJSONDelta MessageContentDeltaType = "input_json_delta"
	MessageContentDeltaTypeThinkingDelta  MessageContentDeltaType = "thinking_delta"
	MessageContentDeltaTypeSignatureDelta MessageContentDeltaType = "signature_delta"
	MessageContentDeltaTypeCitationsDelta MessageContentDeltaType = "citations_delta"
)

type MessageContentDelta struct {
	Type        MessageContentDeltaType `json:"type"`
	Text        string                  `json:"text,omitempty"`
	PartialJSON string                  `json:"partial_json,omitempty"`
	Thinking    string                  `json:"thinking,omitempty"`
	Signature   string                  `json:"signature,omitempty"`
	Citation    *Citation               `json:"citation,omitempty"`
}

type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}

type Thinking struct {
	Type         ThinkingType `json:"type"`
	BudgetTokens int          `json:"budget_tokens"`
}

type ThinkingType string

const (
	ThinkingTypeEnabled  ThinkingType = "enabled"
	ThinkingTypeDisabled ThinkingType = "disabled"
)

type ToolChoice struct {
	Type                   ToolChoiceType `json:"type"`
	Name                   string         `json:"name,omitempty"`
	DisableParallelToolUse bool           `json:"disable_parallel_tool_use,omitempty"`
}

type ToolChoiceType string

const (
	ToolChoiceTypeTool ToolChoiceType = "tool"
	ToolChoiceTypeAuto ToolChoiceType = "auto"
	ToolChoiceTypeNone ToolChoiceType = "none"
	ToolChoiceTypeAny  ToolChoiceType = "any"
)

type Tool struct {
	Type           *ToolType       `json:"type"`
	Name           string          `json:"name"`
	Description    string          `json:"description"`
	InputSchema    json.RawMessage `json:"input_schema"`
	CacheControl   *CacheControl   `json:"cache_control"`
	MaxUses        int             `json:"max_uses"`
	AllowedDomains []string        `json:"allowed_domains"`
	BlockedDomains []string        `json:"blocked_domains"`
	UserLocation   *ToolLocation   `json:"user_location"`
}

// MarshalJSON marshals the Tool to JSON, omitting zero or empty values for slices.
// Anthropic considers the presence of a field as a valid value. When both allowed_domains and blocked_domains
// are set to [], Anthropic still regards this as an error, so we need to handle these fields separately.
//
// reference: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool#domain-filtering
func (t *Tool) MarshalJSON() ([]byte, error) {
	type toolMarshaler struct {
		Type           *ToolType       `json:"type"`
		Name           string          `json:"name"`
		Description    string          `json:"description,omitzero"`   // the description is empty while calling web_search_20250305 tool
		InputSchema    json.RawMessage `json:"input_schema,omitempty"` // the input_schema is null while calling web_search_20250305 tool
		CacheControl   *CacheControl   `json:"cache_control,omitempty"`
		MaxUses        int             `json:"max_uses,omitempty"`
		AllowedDomains []string        `json:"allowed_domains,omitzero"`
		BlockedDomains []string        `json:"blocked_domains,omitzero"`
		UserLocation   *ToolLocation   `json:"user_location,omitempty"`
	}
	aux := toolMarshaler{
		Type:         t.Type,
		Name:         t.Name,
		Description:  t.Description,
		InputSchema:  t.InputSchema,
		CacheControl: t.CacheControl,
		MaxUses:      t.MaxUses,
		UserLocation: t.UserLocation,
	}
	if len(t.AllowedDomains) > 0 {
		aux.AllowedDomains = t.AllowedDomains
	}
	if len(t.BlockedDomains) > 0 {
		aux.BlockedDomains = t.BlockedDomains
	}
	return json.Marshal(&aux)
}

type ToolType string

const (
	ToolTypeCustom        ToolType = "custom"
	ToolTypeWebSearch2025 ToolType = "web_search_20250305"
)

const (
	ToolNameWebSearch = "WebSearch"
)

type ToolLocation struct {
	Type     ToolLocationType `json:"type"`
	City     string           `json:"city,omitempty"`
	Region   string           `json:"region,omitempty"`
	Country  string           `json:"country,omitempty"`
	Timezone string           `json:"timezone,omitempty"`
}

type ToolLocationType string

const (
	ToolLocationTypeApproximate ToolLocationType = "approximate"
)

type Usage struct {
	InputTokens              int64               `json:"input_tokens"`
	OutputTokens             int64               `json:"output_tokens"`
	CacheReadInputTokens     int64               `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64               `json:"cache_creation_input_tokens"`
	CacheCreation            *CacheCreationUsage `json:"cache_creation,omitempty"`
	ServerToolUse            *ServerToolUseUsage `json:"server_tool_use,omitempty"`
}

type CacheCreationUsage struct {
	Ephemeral5MInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1HInputTokens int `json:"ephemeral_1h_input_tokens"`
}

type ServerToolUseUsage struct {
	WebSearchRequests int `json:"web_search_requests"`
}

type MessageStream = iter.Seq2[Event, error]

func NewMessageBuilder() *MessageBuilder {
	return &MessageBuilder{
		message: &Message{
			Type:  MessageTypeMessage,
			Role:  MessageRoleAssistant,
			Usage: &Usage{},
		},
	}
}

type MessageBuilder struct {
	message     *Message
	textBuilder strings.Builder
	jsonBuilder bytes.Buffer
}

func (builder *MessageBuilder) Message() *Message {
	return builder.message
}

func (builder *MessageBuilder) Add(event Event) error {
	switch e := event.(type) {
	case *EventError:
		if e.Error != nil {
			return &Error{
				ContentType: ErrorContentType,
				Inner:       &InnerError{Type: e.Error.ErrType, Message: e.Error.ErrMessage},
			}
		}
	case *EventMessageStart:
		if e.Message != nil {
			builder.message.ID = e.Message.ID
			builder.message.Model = e.Message.Model
			builder.message.Usage.InputTokens = e.Message.Usage.InputTokens
		}
	case *EventMessageDelta:
		if e.Delta != nil {
			builder.message.StopSequence = e.Delta.StopSequence
			builder.message.StopReason = e.Delta.StopReason
		}
		if e.Usage != nil {
			if e.Usage.InputTokens > 0 {
				builder.message.Usage.InputTokens = e.Usage.InputTokens
			}
			if e.Usage.OutputTokens > 0 {
				builder.message.Usage.OutputTokens = e.Usage.OutputTokens
			}
		}
	case *EventContentBlockStart:
		if e.ContentBlock != nil {
			if e.Index >= len(builder.message.Content) {
				builder.message.Content = append(builder.message.Content, &MessageContent{})
				for range e.Index - len(builder.message.Content) {
					builder.message.Content = append(builder.message.Content, &MessageContent{})
				}
			}
			content := builder.message.Content[e.Index]
			content.Type = e.ContentBlock.Type
			switch content.Type {
			case MessageContentTypeToolUse, MessageContentTypeServerToolUse:
				content.ID = e.ContentBlock.ID
				content.Name = e.ContentBlock.Name
			case MessageContentTypeRedactedThinking:
				// Anthropic said that using a special magic string would allow access to the redacted_thinking content,
				// but no matter what I do, I can’t manage to obtain it.
				//
				// reference: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks
				panic("unreachable redacted_thinking")
			case MessageContentTypeWebSearchToolResult:
				content.ToolUseID = e.ContentBlock.ToolUseID
				content.Content = e.ContentBlock.Content
			}
		}
	case *EventContentBlockDelta:
		if e.Delta != nil {
			switch e.Delta.Type {
			case MessageContentDeltaTypeThinkingDelta:
				builder.textBuilder.WriteString(e.Delta.Thinking)
			case MessageContentDeltaTypeSignatureDelta:
				builder.message.Content[e.Index].Signature = e.Delta.Signature
			case MessageContentDeltaTypeTextDelta:
				builder.textBuilder.WriteString(e.Delta.Text)
			case MessageContentDeltaTypeCitationsDelta:
				builder.message.Content[e.Index].Citations = append(builder.message.Content[e.Index].Citations, e.Delta.Citation)
			case MessageContentDeltaTypeInputJSONDelta:
				builder.jsonBuilder.WriteString(e.Delta.PartialJSON)
			}
		}
	case *EventContentBlockStop:
		switch content := builder.message.Content[e.Index]; content.Type {
		case MessageContentTypeThinking:
			content.Thinking = builder.textBuilder.String()
			builder.textBuilder.Reset()
		case MessageContentTypeText:
			content.Text = builder.textBuilder.String()
			builder.textBuilder.Reset()
		case MessageContentTypeToolUse, MessageContentTypeServerToolUse:
			if builder.jsonBuilder.Len() == 0 {
				// For those tools without input_schema.properties (ExitPlanMode)
				//
				// "input_schema": {
				//   "type": "object",
				//   "properties": {},
				//   "additionalProperties": true,
				//   "$schema": "http://json-schema.org/draft-07/schema#"
				// }
				//
				// Anthropic will output an empty object as the input, but OpenRouter may output empty string
				//
				// "tool_calls": [
				//   {
				//   	"index": 0,
				//   	"id": "toolu_01R3kNPHWa518H2x7rVvNZzQ",
				//   	"type": "function",
				//   	"function": {
				//   	  "name": "ExitPlanMode",
				//   	  "arguments": ""
				//      }
				//   }
				// ]
				content.Input = json.RawMessage("{}")
			} else {
				var inputObject map[string]any
				decoder := json.NewDecoder(&builder.jsonBuilder)
				decoder.UseNumber()
				if err := decoder.Decode(&inputObject); err != nil {
					return fmt.Errorf("invalid tool_use json input: %w", err)
				}
				content.Input = json.RawMessage(utils.JSONEncodeString(inputObject))
				builder.jsonBuilder.Reset()
			}
		}
	}
	return nil
}

type EventType string

const (
	EventTypePing              EventType = "ping"
	EventTypeError             EventType = "error"
	EventTypeMessageStart      EventType = "message_start"
	EventTypeMessageDelta      EventType = "message_delta"
	EventTypeMessageStop       EventType = "message_stop"
	EventTypeContentBlockStart EventType = "content_block_start"
	EventTypeContentBlockDelta EventType = "content_block_delta"
	EventTypeContentBlockStop  EventType = "content_block_stop"
)

type Event interface {
	EventType() EventType
}

var (
	_ Event = (*EventPing)(nil)
	_ Event = (*EventError)(nil)
	_ Event = (*EventMessageStart)(nil)
	_ Event = (*EventMessageDelta)(nil)
	_ Event = (*EventMessageStop)(nil)
	_ Event = (*EventContentBlockStart)(nil)
	_ Event = (*EventContentBlockDelta)(nil)
	_ Event = (*EventContentBlockStop)(nil)
)

type (
	EventPing struct {
		Type EventType `json:"type"`
	}
	EventError struct {
		Type  EventType    `json:"type"`
		Error *StreamError `json:"error"`
	}
	EventMessageStart struct {
		Type    EventType `json:"type"`
		Message *Message  `json:"message"`
	}
	EventMessageDelta struct {
		Type  EventType `json:"type"`
		Delta *Message  `json:"delta"`
		Usage *Usage    `json:"usage"`
	}
	EventMessageStop struct {
		Type EventType `json:"type"`
	}
	EventContentBlockStart struct {
		Type         EventType       `json:"type"`
		Index        int             `json:"index"`
		ContentBlock *MessageContent `json:"content_block"`
	}
	EventContentBlockDelta struct {
		Type  EventType            `json:"type"`
		Index int                  `json:"index"`
		Delta *MessageContentDelta `json:"delta"`
	}
	EventContentBlockStop struct {
		Type  EventType `json:"type"`
		Index int       `json:"index"`
	}
)

func (event EventPing) EventType() EventType              { return EventTypePing }
func (event EventError) EventType() EventType             { return EventTypeError }
func (event EventMessageStart) EventType() EventType      { return EventTypeMessageStart }
func (event EventMessageDelta) EventType() EventType      { return EventTypeMessageDelta }
func (event EventMessageStop) EventType() EventType       { return EventTypeMessageStop }
func (event EventContentBlockStart) EventType() EventType { return EventTypeContentBlockStart }
func (event EventContentBlockDelta) EventType() EventType { return EventTypeContentBlockDelta }
func (event EventContentBlockStop) EventType() EventType  { return EventTypeContentBlockStop }
