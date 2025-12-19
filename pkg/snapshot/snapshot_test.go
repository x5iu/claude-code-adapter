package snapshot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/spf13/viper"
	"gopkg.in/yaml.v3"

	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
)

func TestConfig_YAMLToJSON_Full(t *testing.T) {
	yamlData := `
provider: "openrouter"
http:
  port: 2194
options:
  strict: false
  reasoning:
    format: "anthropic-claude-v1"
    effort: "medium"
    delimiter: "/"
  models:
    claude-sonnet-4-20250514: "anthropic/claude-sonnet-4"
    claude-opus-4-1-20250805: "anthropic/claude-opus-4.1"
  context_window_resize_factor: 0.6
  disable_count_tokens_request: false
anthropic:
  enable_pass_through_mode: false
  force_thinking: false
  base_url: "https://api.anthropic.com"
  version: "2023-06-01"
openrouter:
  base_url: "https://openrouter.ai/api"
  model_reasoning_format:
    anthropic/claude-sonnet-4: "anthropic-claude-v1"
    openai/gpt-5: "openai-responses-v1"
  allowed_providers:
    - "anthropic"
    - "google-vertex"
    - "amazon-bedrock"
`
	var cfg Config
	if err := yaml.Unmarshal([]byte(yamlData), &cfg); err != nil {
		t.Fatalf("yaml unmarshal error: %v", err)
	}
	if cfg.Provider != "openrouter" {
		t.Fatalf("unexpected provider: %s", cfg.Provider)
	}
	if cfg.Options == nil || cfg.Options.Reasoning == nil {
		t.Fatalf("options or options.reasoning should not be nil")
	}
	if cfg.Options.Reasoning.Format != "anthropic-claude-v1" || cfg.Options.Reasoning.Effort != "medium" || cfg.Options.Reasoning.Delimiter != "/" {
		t.Fatalf("unexpected reasoning: %#v", cfg.Options.Reasoning)
	}
	if cfg.Anthropic == nil || cfg.Anthropic.BaseURL != "https://api.anthropic.com" || cfg.Anthropic.Version != "2023-06-01" || cfg.Anthropic.ForceThinking != false {
		t.Fatalf("unexpected anthropic: %#v", cfg.Anthropic)
	}
	if cfg.OpenRouter == nil || cfg.OpenRouter.BaseURL != "https://openrouter.ai/api" || len(cfg.OpenRouter.AllowedProviders) != 3 {
		t.Fatalf("unexpected openrouter: %#v", cfg.OpenRouter)
	}
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var cfg2 Config
	if err := json.Unmarshal(b, &cfg2); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if !reflect.DeepEqual(cfg, cfg2) {
		t.Fatalf("config mismatch after YAML->JSON round trip")
	}
}

func TestConfig_YAMLToJSON_Partial(t *testing.T) {
	yamlData := `
provider: "anthropic"
`
	var cfg Config
	if err := yaml.Unmarshal([]byte(yamlData), &cfg); err != nil {
		t.Fatalf("yaml unmarshal error: %v", err)
	}
	if cfg.Provider != "anthropic" {
		t.Fatalf("unexpected provider: %s", cfg.Provider)
	}
	if cfg.Options != nil || cfg.Anthropic != nil || cfg.OpenRouter != nil {
		t.Fatalf("expected nil optional sections")
	}
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var cfg2 Config
	if err := json.Unmarshal(b, &cfg2); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if !reflect.DeepEqual(cfg, cfg2) {
		t.Fatalf("config mismatch after YAML->JSON round trip (partial)")
	}
}

func TestConfig_YAMLToJSON_MapsAndFloats(t *testing.T) {
	yamlData := `
provider: "openrouter"
options:
  models:
    claude-sonnet-4-20250514: "anthropic/claude-sonnet-4"
    claude-opus-4-1-20250805: "anthropic/claude-opus-4.1"
  context_window_resize_factor: 0.6
  disable_count_tokens_request: false
openrouter:
  model_reasoning_format:
    anthropic/claude-sonnet-4: "anthropic-claude-v1"
    openai/gpt-5: "openai-responses-v1"
  allowed_providers:
    - "anthropic"
    - "google-vertex"
    - "amazon-bedrock"
`
	var cfg Config
	if err := yaml.Unmarshal([]byte(yamlData), &cfg); err != nil {
		t.Fatalf("yaml unmarshal error: %v", err)
	}
	if cfg.Options == nil {
		t.Fatalf("options should not be nil")
	}
	if len(cfg.Options.Models) != 2 {
		t.Fatalf("unexpected models len: %d", len(cfg.Options.Models))
	}
	if cfg.Options.Models["claude-sonnet-4-20250514"] != "anthropic/claude-sonnet-4" {
		t.Fatalf("unexpected model mapping for claude-sonnet-4-20250514: %s", cfg.Options.Models["claude-sonnet-4-20250514"])
	}
	if cfg.Options.Models["claude-opus-4-1-20250805"] != "anthropic/claude-opus-4.1" {
		t.Fatalf("unexpected model mapping for claude-opus-4-1-20250805: %s", cfg.Options.Models["claude-opus-4-1-20250805"])
	}
	if cfg.Options.ContextWindowResizeFactor != 0.6 {
		t.Fatalf("unexpected context_window_resize_factor: %v", cfg.Options.ContextWindowResizeFactor)
	}
	if cfg.Options.DisableCountTokensRequest != false {
		t.Fatalf("unexpected disable_count_tokens_request: %v", cfg.Options.DisableCountTokensRequest)
	}
	if cfg.OpenRouter == nil {
		t.Fatalf("openrouter should not be nil")
	}
	if len(cfg.OpenRouter.ModelReasoningFormat) != 2 {
		t.Fatalf("unexpected model_reasoning_format len: %d", len(cfg.OpenRouter.ModelReasoningFormat))
	}
	if cfg.OpenRouter.ModelReasoningFormat["anthropic/claude-sonnet-4"] != "anthropic-claude-v1" {
		t.Fatalf("unexpected model_reasoning_format for anthropic/claude-sonnet-4: %s", cfg.OpenRouter.ModelReasoningFormat["anthropic/claude-sonnet-4"])
	}
	if cfg.OpenRouter.ModelReasoningFormat["openai/gpt-5"] != "openai-responses-v1" {
		t.Fatalf("unexpected model_reasoning_format for openai/gpt-5: %s", cfg.OpenRouter.ModelReasoningFormat["openai/gpt-5"])
	}
	expectedProviders := []string{"anthropic", "google-vertex", "amazon-bedrock"}
	if !reflect.DeepEqual(cfg.OpenRouter.AllowedProviders, expectedProviders) {
		t.Fatalf("unexpected allowed_providers: %#v", cfg.OpenRouter.AllowedProviders)
	}
}

func TestConfig_JSONKeysAndNulls(t *testing.T) {
	cfg := Config{Provider: "anthropic"}
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if m["provider"] != "anthropic" {
		t.Fatalf("unexpected provider in JSON: %v", m["provider"])
	}
	if v, ok := m["options"]; !ok || v != nil {
		t.Fatalf("expected options:null, got: %v (present=%v)", v, ok)
	}
	if v, ok := m["anthropic"]; !ok || v != nil {
		t.Fatalf("expected anthropic:null, got: %v (present=%v)", v, ok)
	}
	if v, ok := m["openrouter"]; !ok || v != nil {
		t.Fatalf("expected openrouter:null, got: %v (present=%v)", v, ok)
	}
}

func TestConfig_YAMLToJSON_Empty(t *testing.T) {
	yamlData := ``
	var cfg Config
	if err := yaml.Unmarshal([]byte(yamlData), &cfg); err != nil {
		t.Fatalf("yaml unmarshal error: %v", err)
	}
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var cfg2 Config
	if err := json.Unmarshal(b, &cfg2); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if !reflect.DeepEqual(cfg, cfg2) {
		t.Fatalf("config mismatch after YAML->JSON round trip (empty)")
	}
}

func TestHeader_MarshalJSON_Basic(t *testing.T) {
	h := Header{
		"Single":    {"one"},
		"Multi":     {"a", "b"},
		"ZeroSlice": {},
		"ZeroNil":   nil,
	}
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if _, ok := m["ZeroSlice"]; ok {
		t.Fatalf("expected ZeroSlice to be omitted")
	}
	if _, ok := m["ZeroNil"]; ok {
		t.Fatalf("expected ZeroNil to be omitted")
	}
	if v, ok := m["Single"]; !ok || v != "one" {
		t.Fatalf("expected Single to be string 'one', got: %v (present=%v)", v, ok)
	}
	arr, ok := m["Multi"].([]any)
	if !ok || len(arr) != 2 || arr[0] != "a" || arr[1] != "b" {
		t.Fatalf("expected Multi to be [\"a\", \"b\"], got: %#v", m["Multi"])
	}
}

func TestHeader_MarshalJSON_Empty(t *testing.T) {
	var h Header
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatalf("json marshal error: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("json unmarshal error: %v", err)
	}
	if len(m) != 0 {
		t.Fatalf("expected empty JSON object, got: %#v", m)
	}
}

func TestViper_Unmarshal_Config_ModelReasoningFormatDots(t *testing.T) {
	yamlData := `
openrouter:
  model_reasoning_format:
    anthropic/claude-sonnet-4.1: "anthropic-claude-v1"
    openai/gpt-5.1: "openai-responses-v1"
`
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(yamlData), 0644); err != nil {
		t.Fatalf("write temp config error: %v", err)
	}
	v := viper.NewWithOptions(viper.KeyDelimiter(delimiter.ViperKeyDelimiter))
	v.SetConfigFile(path)
	v.SetConfigType("yaml")
	if err := v.ReadInConfig(); err != nil {
		t.Fatalf("read config error: %v", err)
	}
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		t.Fatalf("viper unmarshal error: %v", err)
	}
	if cfg.OpenRouter == nil {
		t.Fatalf("openrouter should not be nil")
	}
	if len(cfg.OpenRouter.ModelReasoningFormat) != 2 {
		t.Fatalf("unexpected model_reasoning_format len: %d", len(cfg.OpenRouter.ModelReasoningFormat))
	}
	if cfg.OpenRouter.ModelReasoningFormat["anthropic/claude-sonnet-4.1"] != "anthropic-claude-v1" {
		t.Fatalf("unexpected value for anthropic/claude-sonnet-4.1: %s", cfg.OpenRouter.ModelReasoningFormat["anthropic/claude-sonnet-4.1"])
	}
	if cfg.OpenRouter.ModelReasoningFormat["openai/gpt-5.1"] != "openai-responses-v1" {
		t.Fatalf("unexpected value for openai/gpt-5.1: %s", cfg.OpenRouter.ModelReasoningFormat["openai/gpt-5.1"])
	}
}
