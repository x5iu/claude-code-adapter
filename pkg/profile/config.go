package profile

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/spf13/viper"
	"gopkg.in/yaml.v3"

	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
)

// RootConfig represents the root configuration structure.
type RootConfig struct {
	Profiles map[string]*ProfileConfig `yaml:"profiles" json:"profiles" mapstructure:"profiles"`
	HTTP     *HTTPConfig               `yaml:"http" json:"http" mapstructure:"http"`
	Snapshot string                    `yaml:"snapshot" json:"snapshot" mapstructure:"snapshot"`
}

// ProfileConfig represents a profile configuration in the config file.
// This is similar to Profile but uses the config file structure.
type ProfileConfig struct {
	Models     []string          `yaml:"models" json:"models" mapstructure:"models"`
	Provider   string            `yaml:"provider" json:"provider" mapstructure:"provider"`
	Options    *OptionsConfig    `yaml:"options" json:"options" mapstructure:"options"`
	Anthropic  *AnthropicConfig  `yaml:"anthropic" json:"anthropic" mapstructure:"anthropic"`
	OpenRouter *OpenRouterConfig `yaml:"openrouter" json:"openrouter" mapstructure:"openrouter"`
}

// HTTPConfig contains HTTP server configuration.
type HTTPConfig struct {
	Host string `yaml:"host" json:"host" mapstructure:"host"`
	Port int    `yaml:"port" json:"port" mapstructure:"port"`
}

// envVarRegex matches environment variable references like ${VAR_NAME}
var envVarRegex = regexp.MustCompile(`\$\{([^}]+)\}`)

// expandEnv expands environment variable references in a string.
// Supports ${VAR_NAME} syntax.
func ExpandEnv(s string) string {
	return envVarRegex.ReplaceAllStringFunc(s, func(match string) string {
		// Extract variable name from ${VAR_NAME}
		varName := match[2 : len(match)-1]
		if value, ok := os.LookupEnv(varName); ok {
			return value
		}
		return match // Return original if not found
	})
}

// LoadFromViper loads profiles from a viper instance.
// The profiles section should be structured as:
//
//	profiles:
//	  profile-name:
//	    models: ["pattern*"]
//	    provider: "openrouter"
//	    ...
func LoadFromViper(v *viper.Viper) (*ProfileManager, error) {
	pm := NewProfileManager()
	profilesMap := v.GetStringMap("profiles")
	if len(profilesMap) == 0 {
		return nil, ErrNoProfilesDefined
	}
	// Get profile names in order from the raw config
	// Since viper doesn't preserve order, we need to read the raw config
	profileOrder := getProfileOrder(v)
	for _, name := range profileOrder {
		key := delimiter.ViperKey("profiles", name)
		p := &Profile{
			Name:       name,
			Models:     v.GetStringSlice(delimiter.ViperKey(key, "models")),
			Provider:   v.GetString(delimiter.ViperKey(key, "provider")),
			Options:    loadOptionsConfig(v, delimiter.ViperKey(key, "options")),
			Anthropic:  loadAnthropicConfig(v, delimiter.ViperKey(key, "anthropic")),
			OpenRouter: loadOpenRouterConfig(v, delimiter.ViperKey(key, "openrouter")),
		}
		// Expand environment variables in API keys and URLs
		if p.Anthropic != nil {
			p.Anthropic.APIKey = ExpandEnv(p.Anthropic.APIKey)
			p.Anthropic.BaseURL = ExpandEnv(p.Anthropic.BaseURL)
		}
		if p.OpenRouter != nil {
			p.OpenRouter.APIKey = ExpandEnv(p.OpenRouter.APIKey)
			p.OpenRouter.BaseURL = ExpandEnv(p.OpenRouter.BaseURL)
		}
		pm.AddProfile(p)
	}
	return pm, nil
}

// getProfileOrder attempts to get profile names in their definition order.
// Falls back to map iteration order if order cannot be determined.
func getProfileOrder(v *viper.Viper) []string {
	// Try to get order from the config file
	configFile := v.ConfigFileUsed()
	if configFile != "" {
		if order, err := extractProfileOrderFromFile(configFile); err == nil && len(order) > 0 {
			return order
		}
	}
	// Fallback to map keys (unordered)
	profilesMap := v.GetStringMap("profiles")
	names := make([]string, 0, len(profilesMap))
	for name := range profilesMap {
		names = append(names, name)
	}
	return names
}

// extractProfileOrderFromFile reads the config file and extracts profile names in order.
func extractProfileOrderFromFile(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Profiles yaml.Node `yaml:"profiles"`
	}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw.Profiles.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("profiles is not a mapping")
	}
	var names []string
	// In a mapping node, content alternates between key and value
	for i := 0; i < len(raw.Profiles.Content); i += 2 {
		if raw.Profiles.Content[i].Kind == yaml.ScalarNode {
			names = append(names, raw.Profiles.Content[i].Value)
		}
	}
	return names, nil
}

func loadOptionsConfig(v *viper.Viper, key string) *OptionsConfig {
	if !v.IsSet(key) {
		return nil
	}
	return &OptionsConfig{
		Strict:                     v.GetBool(delimiter.ViperKey(key, "strict")),
		PreventEmptyTextToolResult: v.GetBool(delimiter.ViperKey(key, "prevent_empty_text_tool_result")),
		Reasoning:                  loadReasoningConfig(v, delimiter.ViperKey(key, "reasoning")),
		Models:                     v.GetStringMapString(delimiter.ViperKey(key, "models")),
		ContextWindowResizeFactor:  v.GetFloat64(delimiter.ViperKey(key, "context_window_resize_factor")),
		DisableCountTokensRequest:  v.GetBool(delimiter.ViperKey(key, "disable_count_tokens_request")),
	}
}

func loadReasoningConfig(v *viper.Viper, key string) *ReasoningConfig {
	if !v.IsSet(key) {
		return nil
	}
	return &ReasoningConfig{
		Format:    v.GetString(delimiter.ViperKey(key, "format")),
		Effort:    v.GetString(delimiter.ViperKey(key, "effort")),
		Delimiter: v.GetString(delimiter.ViperKey(key, "delimiter")),
	}
}

func loadAnthropicConfig(v *viper.Viper, key string) *AnthropicConfig {
	if !v.IsSet(key) {
		return nil
	}
	return &AnthropicConfig{
		UseRawRequestBody:              v.GetBool(delimiter.ViperKey(key, "use_raw_request_body")),
		EnablePassThroughMode:          v.GetBool(delimiter.ViperKey(key, "enable_pass_through_mode")),
		DisableInterleavedThinking:     v.GetBool(delimiter.ViperKey(key, "disable_interleaved_thinking")),
		DisableWebSearchBlockedDomains: v.GetBool(delimiter.ViperKey(key, "disable_web_search_blocked_domains")),
		ForceThinking:                  v.GetBool(delimiter.ViperKey(key, "force_thinking")),
		BaseURL:                        v.GetString(delimiter.ViperKey(key, "base_url")),
		APIKey:                         v.GetString(delimiter.ViperKey(key, "api_key")),
		Version:                        v.GetString(delimiter.ViperKey(key, "version")),
	}
}

func loadOpenRouterConfig(v *viper.Viper, key string) *OpenRouterConfig {
	if !v.IsSet(key) {
		return nil
	}
	return &OpenRouterConfig{
		BaseURL:              v.GetString(delimiter.ViperKey(key, "base_url")),
		APIKey:               v.GetString(delimiter.ViperKey(key, "api_key")),
		ModelReasoningFormat: v.GetStringMapString(delimiter.ViperKey(key, "model_reasoning_format")),
		AllowedProviders:     v.GetStringSlice(delimiter.ViperKey(key, "allowed_providers")),
	}
}

// GetHTTPConfig returns the HTTP configuration from viper.
func GetHTTPConfig(v *viper.Viper) *HTTPConfig {
	return &HTTPConfig{
		Host: v.GetString(delimiter.ViperKey("http", "host")),
		Port: v.GetInt(delimiter.ViperKey("http", "port")),
	}
}

// GetSnapshotConfig returns the snapshot configuration from viper.
func GetSnapshotConfig(v *viper.Viper) string {
	return v.GetString("snapshot")
}

// GetOptionsValue safely gets a value from OptionsConfig with a default.
func (o *OptionsConfig) GetStrict() bool {
	if o == nil {
		return false
	}
	return o.Strict
}

// GetPreventEmptyTextToolResult safely gets the value with a default.
func (o *OptionsConfig) GetPreventEmptyTextToolResult() bool {
	if o == nil {
		return false
	}
	return o.PreventEmptyTextToolResult
}

// GetContextWindowResizeFactor safely gets the value with a default.
func (o *OptionsConfig) GetContextWindowResizeFactor() float64 {
	if o == nil {
		return 1.0
	}
	if o.ContextWindowResizeFactor == 0 {
		return 1.0
	}
	return o.ContextWindowResizeFactor
}

// GetDisableCountTokensRequest safely gets the value with a default.
func (o *OptionsConfig) GetDisableCountTokensRequest() bool {
	if o == nil {
		return false
	}
	return o.DisableCountTokensRequest
}

// GetModels safely gets the models map.
func (o *OptionsConfig) GetModels() map[string]string {
	if o == nil || o.Models == nil {
		return make(map[string]string)
	}
	return o.Models
}

// GetReasoningFormat safely gets the reasoning format with a default.
func (o *OptionsConfig) GetReasoningFormat() string {
	if o == nil || o.Reasoning == nil || o.Reasoning.Format == "" {
		return "anthropic-claude-v1"
	}
	return o.Reasoning.Format
}

// GetReasoningEffort safely gets the reasoning effort.
func (o *OptionsConfig) GetReasoningEffort() string {
	if o == nil || o.Reasoning == nil {
		return ""
	}
	return o.Reasoning.Effort
}

// GetReasoningDelimiter safely gets the reasoning delimiter with a default.
func (o *OptionsConfig) GetReasoningDelimiter() string {
	if o == nil || o.Reasoning == nil || o.Reasoning.Delimiter == "" {
		return "/"
	}
	return o.Reasoning.Delimiter
}

// GetBaseURL safely gets the Anthropic base URL with a default.
func (a *AnthropicConfig) GetBaseURL() string {
	if a == nil || a.BaseURL == "" {
		return "https://api.anthropic.com"
	}
	return strings.TrimSuffix(a.BaseURL, "/")
}

// GetVersion safely gets the Anthropic API version with a default.
func (a *AnthropicConfig) GetVersion() string {
	if a == nil || a.Version == "" {
		return "2023-06-01"
	}
	return a.Version
}

// GetAPIKey safely gets the Anthropic API key.
func (a *AnthropicConfig) GetAPIKey() string {
	if a == nil {
		return ""
	}
	return a.APIKey
}

// GetForceThinking safely gets the force thinking flag.
func (a *AnthropicConfig) GetForceThinking() bool {
	if a == nil {
		return false
	}
	return a.ForceThinking
}

// GetDisableInterleavedThinking safely gets the flag.
func (a *AnthropicConfig) GetDisableInterleavedThinking() bool {
	if a == nil {
		return false
	}
	return a.DisableInterleavedThinking
}

// GetEnablePassThroughMode safely gets the flag.
func (a *AnthropicConfig) GetEnablePassThroughMode() bool {
	if a == nil {
		return false
	}
	return a.EnablePassThroughMode
}

// GetUseRawRequestBody safely gets the flag.
func (a *AnthropicConfig) GetUseRawRequestBody() bool {
	if a == nil {
		return false
	}
	return a.UseRawRequestBody
}

// GetDisableWebSearchBlockedDomains safely gets the flag.
func (a *AnthropicConfig) GetDisableWebSearchBlockedDomains() bool {
	if a == nil {
		return false
	}
	return a.DisableWebSearchBlockedDomains
}

// GetBaseURL safely gets the OpenRouter base URL with a default.
func (o *OpenRouterConfig) GetBaseURL() string {
	if o == nil || o.BaseURL == "" {
		return "https://openrouter.ai/api"
	}
	return strings.TrimSuffix(o.BaseURL, "/")
}

// GetAPIKey safely gets the OpenRouter API key.
func (o *OpenRouterConfig) GetAPIKey() string {
	if o == nil {
		return ""
	}
	return o.APIKey
}

// GetModelReasoningFormat safely gets the model reasoning format map.
func (o *OpenRouterConfig) GetModelReasoningFormat() map[string]string {
	if o == nil || o.ModelReasoningFormat == nil {
		return make(map[string]string)
	}
	return o.ModelReasoningFormat
}

// GetAllowedProviders safely gets the allowed providers list.
func (o *OpenRouterConfig) GetAllowedProviders() []string {
	if o == nil {
		return nil
	}
	return o.AllowedProviders
}
