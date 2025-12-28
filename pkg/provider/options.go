package provider

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/x5iu/claude-code-adapter/pkg/profile"
)

// getConfigFromContext retrieves configuration values from the profile in context.
// This function is used by the generated defc code templates.
// The ctx parameter comes from the template's .ctx field.
// Keys should be: "anthropic", "api_key" or "openrouter", "base_url", etc.
func getConfigFromContext(ctx context.Context, keys ...string) string {
	prof, ok := profile.FromContext(ctx)
	if !ok || len(keys) < 2 {
		return ""
	}
	provider := strings.ToLower(keys[0])
	key := strings.ToLower(keys[1])
	switch provider {
	case "anthropic":
		if prof.Anthropic == nil {
			return ""
		}
		switch key {
		case "api_key":
			return prof.Anthropic.GetAPIKey()
		case "base_url":
			return prof.Anthropic.GetBaseURL()
		case "version":
			return prof.Anthropic.GetVersion()
		}
	case "openrouter":
		if prof.OpenRouter == nil {
			return ""
		}
		switch key {
		case "api_key":
			return prof.OpenRouter.GetAPIKey()
		case "base_url":
			return prof.OpenRouter.GetBaseURL()
		}
	case "openai":
		if prof.OpenAI == nil {
			return ""
		}
		switch key {
		case "api_key":
			return prof.OpenAI.GetAPIKey()
		case "base_url":
			return prof.OpenAI.GetBaseURL()
		}
	}
	return ""
}

type RequestOption = func(*http.Request)

func WithQuery(key string, value string) RequestOption {
	return func(req *http.Request) {
		q := req.URL.Query()
		q.Add(key, value)
		req.URL.RawQuery = q.Encode()
	}
}

func WithHeaders(headers http.Header) RequestOption {
	return func(req *http.Request) {
		for k, v := range headers {
			req.Header[k] = v
		}
	}
}

func ReplaceBody(data []byte) RequestOption {
	return func(req *http.Request) {
		if oldBody := req.Body; oldBody != nil {
			oldBody.Close()
		}
		req.ContentLength = int64(len(data))
		req.Body = io.NopCloser(bytes.NewReader(data))
		req.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(data)), nil
		}
	}
}
