package provider

import (
	"context"
	"io"
	"net/http"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
)

//go:generate go tool github.com/x5iu/defc generate --output provider_impl.go --features api/ignore-status,api/get-body,api/retry,api/gzip --func json_encode=utils.JSONEncode --func get_config=getConfig
type Provider interface {
	responseHandler() *ResponseHandler

	// MakeAnthropicMessagesRequest POST retry=2 options(opts) {{ get_config "anthropic" "base_url" }}/v1/messages
	// Content-Type: application/json
	// X-API-Key: {{ get_config "anthropic" "api_key" }}
	// Anthropic-Version: {{ get_config "anthropic" "version" }}
	MakeAnthropicMessagesRequest(
		ctx context.Context,
		req io.Reader,
		opts ...RequestOption,
	) (io.ReadCloser, http.Header, error)

	// GenerateAnthropicMessage POST retry=2 options(opts) {{ get_config "anthropic" "base_url" }}/v1/messages
	// Content-Type: application/json
	// X-API-Key: {{ get_config "anthropic" "api_key" }}
	// Anthropic-Version: {{ get_config "anthropic" "version" }}
	//
	// {{ json_encode .req }}
	GenerateAnthropicMessage(
		ctx context.Context,
		req *anthropic.GenerateMessageRequest,
		opts ...RequestOption,
	) (anthropic.MessageStream, http.Header, error)

	// CountAnthropicTokens POST retry=1 options(opts) {{ get_config "anthropic" "base_url" }}/v1/messages/count_tokens
	// Content-Type: application/json
	// X-API-Key: {{ get_config "anthropic" "api_key" }}
	// Anthropic-Version: {{ get_config "anthropic" "version" }}
	//
	// {{ json_encode .req }}
	CountAnthropicTokens(
		ctx context.Context,
		req *anthropic.CountTokensRequest,
		opts ...RequestOption,
	) (*anthropic.Usage, error)

	// CreateOpenRouterChatCompletion POST retry=2 options(opts) {{ get_config "openrouter" "base_url" }}/v1/chat/completions
	// Content-Type: application/json
	// Authorization: Bearer {{ get_config "openrouter" "api_key" }}
	//
	// {{ json_encode .req }}
	CreateOpenRouterChatCompletion(
		ctx context.Context,
		req *openrouter.CreateChatCompletionRequest,
		opts ...RequestOption,
	) (openrouter.ChatCompletionStream, http.Header, error)
}
