// Package profile provides a profile-based configuration system that allows
// different models to use different provider configurations.
package profile

import (
	"errors"
	"strings"
)

var (
	ErrNoProfileMatched  = errors.New("no profile matched for the given model")
	ErrNoProfilesDefined = errors.New("no profiles defined in configuration")
)

// Profile represents a configuration profile that can be matched against model names.
type Profile struct {
	Name       string            `yaml:"name" json:"name" mapstructure:"name"`
	Models     []string          `yaml:"models" json:"models" mapstructure:"models"`
	Provider   string            `yaml:"provider" json:"provider" mapstructure:"provider"`
	Options    *OptionsConfig    `yaml:"options" json:"options" mapstructure:"options"`
	Anthropic  *AnthropicConfig  `yaml:"anthropic" json:"anthropic" mapstructure:"anthropic"`
	OpenRouter *OpenRouterConfig `yaml:"openrouter" json:"openrouter" mapstructure:"openrouter"`
}

// OptionsConfig contains general options for request processing.
type OptionsConfig struct {
	Strict                     bool              `yaml:"strict" json:"strict" mapstructure:"strict"`
	PreventEmptyTextToolResult bool              `yaml:"prevent_empty_text_tool_result" json:"prevent_empty_text_tool_result" mapstructure:"prevent_empty_text_tool_result"`
	Reasoning                  *ReasoningConfig  `yaml:"reasoning" json:"reasoning" mapstructure:"reasoning"`
	Models                     map[string]string `yaml:"models" json:"models" mapstructure:"models"`
	ContextWindowResizeFactor  float64           `yaml:"context_window_resize_factor" json:"context_window_resize_factor" mapstructure:"context_window_resize_factor"`
	DisableCountTokensRequest  bool              `yaml:"disable_count_tokens_request" json:"disable_count_tokens_request" mapstructure:"disable_count_tokens_request"`
	MinMaxTokens               int               `yaml:"min_max_tokens" json:"min_max_tokens" mapstructure:"min_max_tokens"`
	DisallowedTools            []string          `yaml:"disallowed_tools" json:"disallowed_tools" mapstructure:"disallowed_tools"`
}

// ReasoningConfig contains options for reasoning/thinking mode.
type ReasoningConfig struct {
	Format    string `yaml:"format" json:"format" mapstructure:"format"`
	Effort    string `yaml:"effort" json:"effort" mapstructure:"effort"`
	Delimiter string `yaml:"delimiter" json:"delimiter" mapstructure:"delimiter"`
}

// AnthropicConfig contains Anthropic-specific configuration.
type AnthropicConfig struct {
	UseRawRequestBody              bool   `yaml:"use_raw_request_body" json:"use_raw_request_body" mapstructure:"use_raw_request_body"`
	EnablePassThroughMode          bool   `yaml:"enable_pass_through_mode" json:"enable_pass_through_mode" mapstructure:"enable_pass_through_mode"`
	DisableWebSearchBlockedDomains bool   `yaml:"disable_web_search_blocked_domains" json:"disable_web_search_blocked_domains" mapstructure:"disable_web_search_blocked_domains"`
	ForceThinking                  bool   `yaml:"force_thinking" json:"force_thinking" mapstructure:"force_thinking"`
	BaseURL                        string `yaml:"base_url" json:"base_url" mapstructure:"base_url"`
	APIKey                         string `yaml:"api_key" json:"api_key" mapstructure:"api_key"`
	Version                        string `yaml:"version" json:"version" mapstructure:"version"`
	CountTokensBackend             string `yaml:"count_tokens_backend" json:"count_tokens_backend" mapstructure:"count_tokens_backend"`
}

// OpenRouterConfig contains OpenRouter-specific configuration.
type OpenRouterConfig struct {
	BaseURL              string            `yaml:"base_url" json:"base_url" mapstructure:"base_url"`
	APIKey               string            `yaml:"api_key" json:"api_key" mapstructure:"api_key"`
	ModelReasoningFormat map[string]string `yaml:"model_reasoning_format" json:"model_reasoning_format" mapstructure:"model_reasoning_format"`
	AllowedProviders     []string          `yaml:"allowed_providers" json:"allowed_providers" mapstructure:"allowed_providers"`
}

// ProfileManager manages a collection of profiles and provides model-to-profile matching.
type ProfileManager struct {
	profiles []*Profile // profiles in order of priority
}

// NewProfileManager creates a new empty ProfileManager.
func NewProfileManager() *ProfileManager {
	return &ProfileManager{
		profiles: make([]*Profile, 0),
	}
}

// AddProfile adds a profile to the manager.
func (pm *ProfileManager) AddProfile(p *Profile) {
	pm.profiles = append(pm.profiles, p)
}

// Match finds the first profile that matches the given model name.
// Returns ErrNoProfileMatched if no profile matches.
func (pm *ProfileManager) Match(model string) (*Profile, error) {
	if len(pm.profiles) == 0 {
		return nil, ErrNoProfilesDefined
	}
	for _, p := range pm.profiles {
		for _, pattern := range p.Models {
			if matchPattern(pattern, model) {
				return p, nil
			}
		}
	}
	return nil, ErrNoProfileMatched
}

// Profiles returns all registered profiles.
func (pm *ProfileManager) Profiles() []*Profile {
	return pm.profiles
}

// matchPattern checks if a model name matches a pattern.
// Supports:
// - "*" matches everything
// - "prefix*" matches any model starting with "prefix"
// - exact match for patterns without wildcards
func matchPattern(pattern, model string) bool {
	if pattern == "*" {
		return true
	}
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(model, prefix)
	}
	return pattern == model
}
