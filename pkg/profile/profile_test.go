package profile

import (
	"context"
	"os"
	"testing"
)

func TestMatchPattern(t *testing.T) {
	tests := []struct {
		pattern string
		model   string
		want    bool
	}{
		// Wildcard matches everything
		{"*", "anything", true},
		{"*", "claude-sonnet-4", true},
		{"*", "", true},

		// Prefix matching
		{"claude-*", "claude-sonnet-4", true},
		{"claude-*", "claude-opus-4", true},
		{"claude-*", "gpt-4", false},
		{"anthropic/*", "anthropic/claude-sonnet-4", true},
		{"anthropic/*", "openai/gpt-4", false},

		// Exact matching
		{"claude-sonnet-4", "claude-sonnet-4", true},
		{"claude-sonnet-4", "claude-opus-4", false},
		{"claude-sonnet-4", "claude-sonnet-4-20250514", false},

		// Edge cases
		{"", "", true},
		{"", "anything", false},
		{"prefix*", "prefix", true}, // prefix* matches "prefix" exactly too
	}

	for _, tt := range tests {
		t.Run(tt.pattern+"_"+tt.model, func(t *testing.T) {
			got := matchPattern(tt.pattern, tt.model)
			if got != tt.want {
				t.Errorf("matchPattern(%q, %q) = %v, want %v", tt.pattern, tt.model, got, tt.want)
			}
		})
	}
}

func TestProfileManager_Match(t *testing.T) {
	pm := NewProfileManager()

	// Add profiles in order
	pm.AddProfile(&Profile{
		Name:     "anthropic-claude",
		Models:   []string{"claude-*", "anthropic/*"},
		Provider: "anthropic",
	})
	pm.AddProfile(&Profile{
		Name:     "openrouter-gpt",
		Models:   []string{"gpt-*", "openai/*"},
		Provider: "openrouter",
	})
	pm.AddProfile(&Profile{
		Name:     "gemini",
		Models:   []string{"google/*", "gemini-*"},
		Provider: "openrouter",
	})

	tests := []struct {
		model       string
		wantProfile string
		wantErr     error
	}{
		{"claude-sonnet-4", "anthropic-claude", nil},
		{"claude-opus-4", "anthropic-claude", nil},
		{"anthropic/claude-sonnet-4", "anthropic-claude", nil},
		{"gpt-4", "openrouter-gpt", nil},
		{"openai/gpt-4-turbo", "openrouter-gpt", nil},
		{"google/gemini-pro", "gemini", nil},
		{"gemini-2.0-flash", "gemini", nil},
		{"unknown-model", "", ErrNoProfileMatched},
		{"llama-3", "", ErrNoProfileMatched},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got, err := pm.Match(tt.model)
			if err != tt.wantErr {
				t.Errorf("Match(%q) error = %v, want %v", tt.model, err, tt.wantErr)
				return
			}
			if tt.wantErr == nil && got.Name != tt.wantProfile {
				t.Errorf("Match(%q) = %q, want %q", tt.model, got.Name, tt.wantProfile)
			}
		})
	}
}

func TestProfileManager_MatchPriority(t *testing.T) {
	pm := NewProfileManager()

	// Add a catch-all profile first
	pm.AddProfile(&Profile{
		Name:     "catch-all",
		Models:   []string{"*"},
		Provider: "openrouter",
	})
	// Add a more specific profile second
	pm.AddProfile(&Profile{
		Name:     "claude",
		Models:   []string{"claude-*"},
		Provider: "anthropic",
	})

	// The catch-all should match first since it was added first
	got, err := pm.Match("claude-sonnet-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "catch-all" {
		t.Errorf("expected catch-all to match first, got %q", got.Name)
	}
}

func TestProfileManager_EmptyProfiles(t *testing.T) {
	pm := NewProfileManager()

	_, err := pm.Match("any-model")
	if err != ErrNoProfilesDefined {
		t.Errorf("expected ErrNoProfilesDefined, got %v", err)
	}
}

func TestContext(t *testing.T) {
	ctx := context.Background()

	// Test FromContext returns false when no profile is set
	_, ok := FromContext(ctx)
	if ok {
		t.Error("expected FromContext to return false for empty context")
	}

	// Test WithProfile and FromContext
	profile := &Profile{
		Name:     "test-profile",
		Provider: "anthropic",
	}
	ctx = WithProfile(ctx, profile)

	got, ok := FromContext(ctx)
	if !ok {
		t.Error("expected FromContext to return true after WithProfile")
	}
	if got != profile {
		t.Error("expected same profile instance from context")
	}
	if got.Name != "test-profile" {
		t.Errorf("expected profile name 'test-profile', got %q", got.Name)
	}
}

func TestMustFromContext(t *testing.T) {
	// Test panic when no profile
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected MustFromContext to panic on empty context")
		}
	}()

	ctx := context.Background()
	MustFromContext(ctx)
}

func TestMustFromContext_Success(t *testing.T) {
	profile := &Profile{Name: "test"}
	ctx := WithProfile(context.Background(), profile)

	got := MustFromContext(ctx)
	if got.Name != "test" {
		t.Errorf("expected profile name 'test', got %q", got.Name)
	}
}

func TestExpandEnv(t *testing.T) {
	// Set test environment variables
	os.Setenv("TEST_API_KEY", "sk-test-123")
	os.Setenv("TEST_URL", "https://example.com")
	defer func() {
		os.Unsetenv("TEST_API_KEY")
		os.Unsetenv("TEST_URL")
	}()

	tests := []struct {
		input string
		want  string
	}{
		{"${TEST_API_KEY}", "sk-test-123"},
		{"${TEST_URL}/v1", "https://example.com/v1"},
		{"prefix_${TEST_API_KEY}_suffix", "prefix_sk-test-123_suffix"},
		{"${UNDEFINED_VAR}", "${UNDEFINED_VAR}"},
		{"no variables", "no variables"},
		{"", ""},
		{"${TEST_API_KEY}${TEST_URL}", "sk-test-123https://example.com"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := ExpandEnv(tt.input)
			if got != tt.want {
				t.Errorf("ExpandEnv(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestOptionsConfig_Getters(t *testing.T) {
	// Test nil options
	var nilOpts *OptionsConfig
	if nilOpts.GetStrict() != false {
		t.Error("GetStrict on nil should return false")
	}
	if nilOpts.GetContextWindowResizeFactor() != 1.0 {
		t.Error("GetContextWindowResizeFactor on nil should return 1.0")
	}
	if nilOpts.GetReasoningFormat() != "anthropic-claude-v1" {
		t.Error("GetReasoningFormat on nil should return default")
	}
	if nilOpts.GetReasoningDelimiter() != "/" {
		t.Error("GetReasoningDelimiter on nil should return /")
	}

	// Test zero value
	opts := &OptionsConfig{}
	if opts.GetContextWindowResizeFactor() != 1.0 {
		t.Error("GetContextWindowResizeFactor with zero value should return 1.0")
	}

	// Test with values
	opts = &OptionsConfig{
		Strict:                    true,
		ContextWindowResizeFactor: 0.6,
		Reasoning: &ReasoningConfig{
			Format:    "openai-responses-v1",
			Delimiter: "::",
		},
	}
	if opts.GetStrict() != true {
		t.Error("GetStrict should return true")
	}
	if opts.GetContextWindowResizeFactor() != 0.6 {
		t.Error("GetContextWindowResizeFactor should return 0.6")
	}
	if opts.GetReasoningFormat() != "openai-responses-v1" {
		t.Error("GetReasoningFormat should return set value")
	}
	if opts.GetReasoningDelimiter() != "::" {
		t.Error("GetReasoningDelimiter should return set value")
	}
}

func TestAnthropicConfig_Getters(t *testing.T) {
	// Test nil config
	var nilCfg *AnthropicConfig
	if nilCfg.GetBaseURL() != "https://api.anthropic.com" {
		t.Error("GetBaseURL on nil should return default")
	}
	if nilCfg.GetVersion() != "2023-06-01" {
		t.Error("GetVersion on nil should return default")
	}
	if nilCfg.GetAPIKey() != "" {
		t.Error("GetAPIKey on nil should return empty string")
	}

	// Test with values
	cfg := &AnthropicConfig{
		BaseURL: "https://custom.api.com/",
		Version: "2024-01-01",
		APIKey:  "sk-custom",
	}
	if cfg.GetBaseURL() != "https://custom.api.com" {
		t.Errorf("GetBaseURL should trim trailing slash, got %q", cfg.GetBaseURL())
	}
	if cfg.GetVersion() != "2024-01-01" {
		t.Error("GetVersion should return set value")
	}
	if cfg.GetAPIKey() != "sk-custom" {
		t.Error("GetAPIKey should return set value")
	}
}

func TestOpenRouterConfig_Getters(t *testing.T) {
	// Test nil config
	var nilCfg *OpenRouterConfig
	if nilCfg.GetBaseURL() != "https://openrouter.ai/api" {
		t.Error("GetBaseURL on nil should return default")
	}
	if nilCfg.GetAPIKey() != "" {
		t.Error("GetAPIKey on nil should return empty string")
	}
	if nilCfg.GetModelReasoningFormat() == nil {
		t.Error("GetModelReasoningFormat on nil should return empty map, not nil")
	}
	if nilCfg.GetAllowedProviders() != nil {
		t.Error("GetAllowedProviders on nil should return nil")
	}

	// Test with values
	cfg := &OpenRouterConfig{
		BaseURL:              "https://custom.openrouter.com/api/",
		APIKey:               "or-custom",
		ModelReasoningFormat: map[string]string{"model": "format"},
		AllowedProviders:     []string{"anthropic"},
	}
	if cfg.GetBaseURL() != "https://custom.openrouter.com/api" {
		t.Errorf("GetBaseURL should trim trailing slash, got %q", cfg.GetBaseURL())
	}
}
