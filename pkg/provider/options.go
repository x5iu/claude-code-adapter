package provider

import (
	"bytes"
	"io"
	"net/http"

	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
)

func init() {
	viper.SetDefault(delimiter.ViperKey("provider"), "openrouter")
	viper.SetDefault(delimiter.ViperKey("openrouter", "base_url"), "https://openrouter.ai/api")
	viper.MustBindEnv(delimiter.ViperKey("openrouter", "base_url"), "OPENROUTER_BASE_URL")
	viper.MustBindEnv(delimiter.ViperKey("openrouter", "api_key"), "OPENROUTER_API_KEY")
	viper.SetDefault(delimiter.ViperKey("anthropic", "base_url"), "https://api.anthropic.com")
	viper.SetDefault(delimiter.ViperKey("anthropic", "version"), "2023-06-01")
	viper.MustBindEnv(delimiter.ViperKey("anthropic", "base_url"), "ANTHROPIC_BASE_URL")
	viper.MustBindEnv(delimiter.ViperKey("anthropic", "api_key"), "ANTHROPIC_API_KEY")
	viper.MustBindEnv(delimiter.ViperKey("anthropic", "version"), "ANTHROPIC_VERSION")
}

func getConfig(key ...string) string {
	return viper.GetString(delimiter.ViperKey(key...))
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
