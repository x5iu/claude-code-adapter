package snapshot

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
)

type Recorder interface {
	io.Closer
	Record(snapshot *Snapshot) error
}

func NopRecorder() Recorder {
	return nopRecorder{}
}

type nopRecorder struct{}

func (nopRecorder) Close() error                    { return nil }
func (nopRecorder) Record(snapshot *Snapshot) error { return nil }

type Snapshot struct {
	RequestTime        time.Time                               `json:"request_time"`
	FinishTime         time.Time                               `json:"finish_time"`
	Version            string                                  `json:"version"`
	RequestID          string                                  `json:"request_id"`
	StatusCode         int                                     `json:"status_code"`
	Provider           string                                  `json:"provider"`
	Profile            string                                  `json:"profile,omitempty"`
	Config             *Config                                 `json:"config,omitempty"`
	Error              *Error                                  `json:"error,omitempty"`
	AnthropicRequest   *anthropic.GenerateMessageRequest       `json:"anthropic_request,omitempty"`
	AnthropicResponse  *anthropic.Message                      `json:"anthropic_response,omitempty"`
	OpenRouterRequest  *openrouter.CreateChatCompletionRequest `json:"openrouter_request,omitempty"`
	OpenRouterResponse *openrouter.ChatCompletion              `json:"openrouter_response,omitempty"`
	RequestHeader      Header                                  `json:"request_header,omitempty"`
	ResponseHeader     Header                                  `json:"response_header,omitempty"`
}

type Error struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Source  string `json:"source,omitempty"`
}

type Config struct {
	Provider   string            `yaml:"provider" json:"provider" mapstructure:"provider"`
	Options    *OptionsConfig    `yaml:"options" json:"options" mapstructure:"options"`
	Anthropic  *AnthropicConfig  `yaml:"anthropic" json:"anthropic" mapstructure:"anthropic"`
	OpenRouter *OpenRouterConfig `yaml:"openrouter" json:"openrouter" mapstructure:"openrouter"`
}

type OptionsConfig struct {
	Strict                     bool              `yaml:"strict" json:"strict" mapstructure:"strict"`
	PreventEmptyTextToolResult bool              `yaml:"prevent_empty_text_tool_result" json:"prevent_empty_text_tool_result" mapstructure:"prevent_empty_text_tool_result"`
	Reasoning                  *ReasoningConfig  `yaml:"reasoning" json:"reasoning" mapstructure:"reasoning"`
	Models                     map[string]string `yaml:"models" json:"models" mapstructure:"models"`
	ContextWindowResizeFactor  float64           `yaml:"context_window_resize_factor" json:"context_window_resize_factor" mapstructure:"context_window_resize_factor"`
	DisableCountTokensRequest  bool              `yaml:"disable_count_tokens_request" json:"disable_count_tokens_request" mapstructure:"disable_count_tokens_request"`
	MinMaxTokens               int               `yaml:"min_max_tokens" json:"min_max_tokens" mapstructure:"min_max_tokens"`
}

type ReasoningConfig struct {
	Format    string `yaml:"format" json:"format" mapstructure:"format"`
	Effort    string `yaml:"effort" json:"effort" mapstructure:"effort"`
	Delimiter string `yaml:"delimiter" json:"delimiter" mapstructure:"delimiter"`
}

type AnthropicConfig struct {
	UseRawRequestBody              bool   `yaml:"use_raw_request_body" json:"use_raw_request_body" mapstructure:"use_raw_request_body"`
	EnablePassThroughMode          bool   `yaml:"enable_pass_through_mode" json:"enable_pass_through_mode" mapstructure:"enable_pass_through_mode"`
	DisableInterleavedThinking     bool   `yaml:"disable_interleaved_thinking" json:"disable_interleaved_thinking" mapstructure:"disable_interleaved_thinking"`
	DisableWebSearchBlockedDomains bool   `yaml:"disable_web_search_blocked_domains" json:"disable_web_search_blocked_domains" mapstructure:"disable_web_search_blocked_domains"`
	ForceThinking                  bool   `yaml:"force_thinking" json:"force_thinking" mapstructure:"force_thinking"`
	BaseURL                        string `yaml:"base_url" json:"base_url" mapstructure:"base_url"`
	Version                        string `yaml:"version" json:"version" mapstructure:"version"`
}

type OpenRouterConfig struct {
	BaseURL              string            `yaml:"base_url" json:"base_url" mapstructure:"base_url"`
	ModelReasoningFormat map[string]string `yaml:"model_reasoning_format" json:"model_reasoning_format" mapstructure:"model_reasoning_format"`
	AllowedProviders     []string          `yaml:"allowed_providers" json:"allowed_providers" mapstructure:"allowed_providers"`
}

type Header http.Header

func (h Header) MarshalJSON() ([]byte, error) {
	x := make(map[string]any, len(h))
	for k, vv := range h {
		switch len(vv) {
		case 0:
			continue
		case 1:
			x[k] = vv[0]
		default:
			x[k] = vv
		}
	}
	return json.Marshal(x)
}
