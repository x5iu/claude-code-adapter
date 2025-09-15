package openrouter

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"testing"

	"github.com/samber/lo"
)

func TestChatCompletionBuilder_AggregatesMetadataAndUsage(t *testing.T) {
	b := NewChatCompletionBuilder()

	chunk1 := &ChatCompletionChunk{
		ID:       "test-id",
		Provider: "anthropic",
		Model:    "anthropic/claude-opus-4.1",
		Created:  1234567890,
		Object:   "chat.completion.chunk",
		Choices: []*ChatCompletionChunkChoice{{
			Index: 0,
			Delta: &ChatCompletionChunkChoiceDelta{
				Role:    ChatCompletionMessageRoleAssistant,
				Content: "Hel",
			},
		}},
		Usage: &ChatCompletionUsage{
			PromptTokens:     22,
			CompletionTokens: 1336,
			TotalTokens:      1358,
			Cost:             json.Number("0.013253625"),
			IsByok:           false,
			PromptTokensDetails: &ChatCompletionPromptTokensDetails{
				CachedTokens: 0,
				AudioTokens:  0,
			},
			CostDetails: &ChatCompletionCostDetails{
				UpstreamInferenceCost:            nil,
				UpstreamInferencePromptCost:      json.Number("0.0000275"),
				UpstreamInferenceCompletionsCost: json.Number("0.01336"),
			},
			CompletionTokensDetails: &ChatCompletionCompletionTokensDetails{
				ReasoningTokens: 0,
				ImageTokens:     0,
			},
		},
	}

	b.Add(chunk1)

	chunk2 := &ChatCompletionChunk{
		ID:       "test-id",
		Provider: "anthropic",
		Model:    "anthropic/claude-opus-4.1",
		Created:  1234567890,
		Object:   "chat.completion.chunk",
		Choices: []*ChatCompletionChunkChoice{{
			Index:        0,
			Delta:        &ChatCompletionChunkChoiceDelta{Content: "lo"},
			FinishReason: ChatCompletionFinishReasonStop,
		}},
	}

	b.Add(chunk2)

	c := b.Build()

	if c.ID != "test-id" || c.Provider != "anthropic" || c.Model == "" || c.Object == "" {
		t.Fatalf("metadata mismatch: %+v", c)
	}
	if len(c.Choices) != 1 || c.Choices[0] == nil || c.Choices[0].Message == nil || c.Choices[0].Message.Content == nil {
		t.Fatalf("invalid choices: %+v", c.Choices)
	}
	if c.Choices[0].Message.Content.Text != "Hello" {
		t.Fatalf("content not accumulated: %q", c.Choices[0].Message.Content.Text)
	}
	if c.Choices[0].FinishReason != ChatCompletionFinishReasonStop {
		t.Fatalf("finish reason mismatch: %v", c.Choices[0].FinishReason)
	}

	if c.Usage == nil {
		t.Fatalf("usage is nil")
	}
	if c.Usage.PromptTokens != 22 || c.Usage.CompletionTokens != 1336 || c.Usage.TotalTokens != 1358 {
		t.Fatalf("tokens mismatch: %+v", c.Usage)
	}
	if cf, _ := c.Usage.Cost.Float64(); math.Abs(cf-0.013253625) > 1e-12 {
		t.Fatalf("cost mismatch: %v", c.Usage.Cost)
	}
	if c.Usage.IsByok {
		t.Fatalf("is_byok mismatch: %v", c.Usage.IsByok)
	}
	if c.Usage.PromptTokensDetails == nil || c.Usage.PromptTokensDetails.CachedTokens != 0 || c.Usage.PromptTokensDetails.AudioTokens != 0 {
		t.Fatalf("prompt_tokens_details mismatch: %+v", c.Usage.PromptTokensDetails)
	}
	if c.Usage.CostDetails == nil {
		t.Fatalf("cost_details missing")
	}
	if c.Usage.CostDetails.UpstreamInferenceCost != nil {
		t.Fatalf("expected nil upstream_inference_cost: %+v", c.Usage.CostDetails.UpstreamInferenceCost)
	}
	if v, _ := c.Usage.CostDetails.UpstreamInferencePromptCost.Float64(); math.Abs(v-0.0000275) > 1e-12 {
		t.Fatalf("upstream_inference_prompt_cost mismatch: %v", c.Usage.CostDetails.UpstreamInferencePromptCost)
	}
	if v2, _ := c.Usage.CostDetails.UpstreamInferenceCompletionsCost.Float64(); math.Abs(v2-0.01336) > 1e-12 {
		t.Fatalf("upstream_inference_completions_cost mismatch: %v", c.Usage.CostDetails.UpstreamInferenceCompletionsCost)
	}
	if c.Usage.CompletionTokensDetails == nil || c.Usage.CompletionTokensDetails.ReasoningTokens != 0 || c.Usage.CompletionTokensDetails.ImageTokens != 0 {
		t.Fatalf("completion_tokens_details mismatch: %+v", c.Usage.CompletionTokensDetails)
	}
}

func TestChatCompletionBuilder_TotalTokensFallback(t *testing.T) {
	b := NewChatCompletionBuilder()

	chunk := &ChatCompletionChunk{
		ID:       "x",
		Provider: "anthropic",
		Model:    "m",
		Created:  1,
		Object:   "chat.completion.chunk",
		Choices: []*ChatCompletionChunkChoice{{
			Index: 0,
			Delta: &ChatCompletionChunkChoiceDelta{Content: "ok"},
		}},
		Usage: &ChatCompletionUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 0},
	}

	b.Add(chunk)
	c := b.Build()

	if c.Usage == nil || c.Usage.TotalTokens != 15 {
		t.Fatalf("total_tokens fallback failed: %+v", c.Usage)
	}
}

func TestChatCompletionBuilder_MultipleChoicesOutOfOrder(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 1, Delta: &ChatCompletionChunkChoiceDelta{Content: "B"}}}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "A"}}}})
	c := b.Build()
	if len(c.Choices) != 2 {
		t.Fatalf("choices len: %d", len(c.Choices))
	}
	if c.Choices[0].Message.Content.Text != "A" || c.Choices[1].Message.Content.Text != "B" {
		t.Fatalf("out of order aggregation failed: %+v", c.Choices)
	}
}

func TestChatCompletionBuilder_FinishReasonFirstWins(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, FinishReason: ChatCompletionFinishReasonLength}}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, FinishReason: ChatCompletionFinishReasonStop}}})
	c := b.Build()
	if c.Choices[0].FinishReason != ChatCompletionFinishReasonLength {
		t.Fatalf("finish reason overwritten: %v", c.Choices[0].FinishReason)
	}
}

func TestChatCompletionMessageBuilder_RoleAndRefusalAccumulation(t *testing.T) {
	b := &ChatCompletionMessageBuilder{}
	b.Add(&ChatCompletionChunkChoiceDelta{Role: ChatCompletionMessageRoleAssistant, Refusal: lo.ToPtr("R1")})
	b.Add(&ChatCompletionChunkChoiceDelta{Refusal: lo.ToPtr("R2")})
	m := b.Build()
	if m.Role != ChatCompletionMessageRoleAssistant {
		t.Fatalf("role mismatch: %v", m.Role)
	}
	if m.Refusal == nil || *m.Refusal != "R1R2" {
		t.Fatalf("refusal accumulation: %v", m.Refusal)
	}
}

func TestToolCalls_AccumulationAndExpansion(t *testing.T) {
	mb := &ChatCompletionMessageBuilder{}
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 0, ID: "t1", Type: ChatCompletionMessageToolCallTypeFunction, Function: &ChatCompletionMessageToolCallFunction{Name: "g", Arguments: ""}}}})
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 1, ID: "t2", Type: ChatCompletionMessageToolCallTypeFunction, Function: &ChatCompletionMessageToolCallFunction{Name: "f", Arguments: "x"}}}})
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 1, Function: &ChatCompletionMessageToolCallFunction{Arguments: "y"}}}})
	m := mb.Build()
	if len(m.ToolCalls) != 2 {
		t.Fatalf("toolcalls len: %d", len(m.ToolCalls))
	}
	if m.ToolCalls[1].Function == nil || m.ToolCalls[1].Function.Name != "f" || m.ToolCalls[1].Function.Arguments != "xy" {
		t.Fatalf("toolcalls accumulation failed: %+v", m.ToolCalls[1])
	}
}

func TestErrorTypeMapping(t *testing.T) {
	e := &Error{}
	e.Inner.Code = 429
	if e.Type() != "overloaded_error" {
		t.Fatalf("429 type: %s", e.Type())
	}
	e.Inner.Code = 400
	if e.Type() != "invalid_request_error" {
		t.Fatalf("4xx type: %s", e.Type())
	}
	e.Inner.Code = 500
	if e.Type() != "internal_server_error" {
		t.Fatalf("5xx type: %s", e.Type())
	}
	e.Inner.Code = 0
	if e.Type() != "unknown_error" {
		t.Fatalf("default type: %s", e.Type())
	}
}

func TestResponseFormat_Unmarshal(t *testing.T) {
	var f ChatCompletionResponseFormat
	if err := json.Unmarshal([]byte("\"text\""), &f); err != nil || f.Type != ChatCompletionResponseFormatTypeText {
		t.Fatalf("text format fail: %v %#v", err, f)
	}
	if err := json.Unmarshal([]byte("{\"type\":\"json_object\"}"), &f); err != nil || f.Type != ChatCompletionResponseFormatTypeJSONObject {
		t.Fatalf("json_object fail: %v %#v", err, f)
	}
	if err := json.Unmarshal([]byte("{\"type\":\"bad\"}"), &f); err == nil {
		t.Fatalf("expect error for bad type")
	}
	if err := json.Unmarshal([]byte("\"json\""), &f); err == nil {
		t.Fatalf("expect error for invalid string type")
	}
}

func TestStop_Unmarshal(t *testing.T) {
	var s ChatCompletionStop
	if err := json.Unmarshal([]byte("\"eos\""), &s); err != nil || len(s) != 1 || s[0] != "eos" {
		t.Fatalf("string stop fail: %v %#v", err, s)
	}
	if err := json.Unmarshal([]byte("null"), &s); err != nil || len(s) != 1 || s[0] != "eos" {
		t.Fatalf("null should keep existing value: %v %#v", err, s)
	}
	var s2 ChatCompletionStop
	if err := json.Unmarshal([]byte("null"), &s2); err != nil || len(s2) != 0 {
		t.Fatalf("null into zero value should be empty: %v %#v", err, s2)
	}
}

func TestWithProviderPreferenceAndIdentity(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}
	body := []byte(`{"messages":[],"model":"m","stream":true}`)
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	pref := &ProviderPreference{Only: []string{"anthropic"}}
	WithProviderPreference(pref)(req)
	b1, _ := io.ReadAll(req.Body)
	var r1 CreateChatCompletionRequest
	_ = json.Unmarshal(b1, &r1)
	if r1.Provider == nil || len(r1.Provider.Only) != 1 || r1.Provider.Only[0] != "anthropic" {
		t.Fatalf("provider not set: %#v", r1.Provider)
	}
	if req.ContentLength != int64(len(b1)) {
		t.Fatalf("content-length mismatch: %d vs %d", req.ContentLength, len(b1))
	}
	rg, _ := req.GetBody()
	b2, _ := io.ReadAll(rg)
	if string(b1) != string(b2) {
		t.Fatalf("getbody mismatch")
	}
	WithIdentity("r", "t")(req)
	if req.Header.Get("HTTP-Referer") != "r" || req.Header.Get("X-Title") != "t" {
		t.Fatalf("identity headers not set")
	}
}

func TestUsageMergeRules(t *testing.T) {
	b := NewChatCompletionBuilder()
	f := json.Number("0")
	chunk1 := &ChatCompletionChunk{
		Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "x"}}},
		Usage: &ChatCompletionUsage{
			PromptTokens:            5,
			CompletionTokens:        1,
			TotalTokens:             6,
			Cost:                    json.Number("0.1"),
			IsByok:                  false,
			PromptTokensDetails:     &ChatCompletionPromptTokensDetails{CachedTokens: 1, AudioTokens: 2},
			CostDetails:             &ChatCompletionCostDetails{UpstreamInferenceCost: nil, UpstreamInferencePromptCost: json.Number("1.0"), UpstreamInferenceCompletionsCost: json.Number("2.0")},
			CompletionTokensDetails: &ChatCompletionCompletionTokensDetails{ReasoningTokens: 3, ImageTokens: 4},
		},
	}
	b.Add(chunk1)
	chunk2 := &ChatCompletionChunk{
		Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}},
		Usage: &ChatCompletionUsage{
			PromptTokens:            0,
			CompletionTokens:        7,
			TotalTokens:             0,
			Cost:                    json.Number("0"),
			IsByok:                  true,
			PromptTokensDetails:     &ChatCompletionPromptTokensDetails{CachedTokens: 3, AudioTokens: 9},
			CostDetails:             &ChatCompletionCostDetails{UpstreamInferenceCost: &f, UpstreamInferencePromptCost: json.Number("1.5"), UpstreamInferenceCompletionsCost: json.Number("2.5")},
			CompletionTokensDetails: &ChatCompletionCompletionTokensDetails{ReasoningTokens: 5, ImageTokens: 6},
		},
	}
	b.Add(chunk2)
	c := b.Build()
	if c.Usage.PromptTokens != 5 {
		t.Fatalf("prompt tokens overwrite: %d", c.Usage.PromptTokens)
	}
	if c.Usage.CompletionTokens != 7 {
		t.Fatalf("completion tokens not updated: %d", c.Usage.CompletionTokens)
	}
	if c.Usage.TotalTokens != 6 {
		t.Fatalf("total tokens unexpected: %d", c.Usage.TotalTokens)
	}
	if cf, _ := c.Usage.Cost.Float64(); math.Abs(cf-0.1) > 1e-9 {
		t.Fatalf("cost updated unexpectedly: %v", c.Usage.Cost)
	}
	if !c.Usage.IsByok {
		t.Fatalf("is_byok OR failed")
	}
	if c.Usage.PromptTokensDetails == nil || c.Usage.PromptTokensDetails.CachedTokens != 3 || c.Usage.PromptTokensDetails.AudioTokens != 9 {
		t.Fatalf("prompt tokens details merge failed: %+v", c.Usage.PromptTokensDetails)
	}
	if c.Usage.CostDetails == nil || c.Usage.CostDetails.UpstreamInferenceCost == nil {
		t.Fatalf("cost details merge failed: %+v", c.Usage.CostDetails)
	}
	if v, _ := c.Usage.CostDetails.UpstreamInferenceCost.Float64(); v != 0.0 {
		t.Fatalf("upstream_inference_cost mismatch: %v", v)
	}
	if v1, _ := c.Usage.CostDetails.UpstreamInferencePromptCost.Float64(); math.Abs(v1-1.5) > 1e-12 {
		t.Fatalf("prompt_cost mismatch: %v", v1)
	}
	if v2, _ := c.Usage.CostDetails.UpstreamInferenceCompletionsCost.Float64(); math.Abs(v2-2.5) > 1e-12 {
		t.Fatalf("completions_cost mismatch: %v", v2)
	}
	if c.Usage.CompletionTokensDetails == nil || c.Usage.CompletionTokensDetails.ReasoningTokens != 5 || c.Usage.CompletionTokensDetails.ImageTokens != 6 {
		t.Fatalf("completion tokens details merge failed: %+v", c.Usage.CompletionTokensDetails)
	}
}

func TestGetTokensWhenUsageNil(t *testing.T) {
	c := ChatCompletion{}
	if c.GetPromptTokens() != 0 || c.GetCompletionTokens() != 0 {
		t.Fatalf("nil usage getters not zero: %+v", c.Usage)
	}
}

func TestChatCompletionBuilder_NilAndEmptyChunks(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(nil)
	b.Add(&ChatCompletionChunk{})
	c := b.Build()
	if c == nil || len(c.Choices) != 0 {
		t.Fatalf("unexpected build result: %#v", c)
	}
}

func TestChatCompletionBuilder_MetadataFirstWins(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{ID: "a", Provider: "p1", Model: "m1", Created: 1, Object: "o", Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "x"}}}})
	b.Add(&ChatCompletionChunk{ID: "b", Provider: "p2", Model: "m2", Created: 2, Object: "o2", Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "y"}}}})
	c := b.Build()
	if c.ID != "a" || c.Provider != "p1" || c.Model != "m1" || c.Created != 1 || c.Object != "o" {
		t.Fatalf("metadata not first-wins: %#v", c)
	}
	if c.Choices[0].Message.Content.Text != "xy" {
		t.Fatalf("content not appended: %q", c.Choices[0].Message.Content.Text)
	}
}

func TestChatCompletionBuilder_ChoiceGapsAndNonZeroStart(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 2, Delta: &ChatCompletionChunkChoiceDelta{Content: "C"}}}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "A"}}}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 2, Delta: &ChatCompletionChunkChoiceDelta{Content: "c"}}}})
	c := b.Build()
	if len(c.Choices) != 3 {
		t.Fatalf("choices len: %d", len(c.Choices))
	}
	if c.Choices[0].Message.Content.Text != "A" {
		t.Fatalf("index0 mismatch: %q", c.Choices[0].Message.Content.Text)
	}
	if c.Choices[2].Message.Content.Text != "Cc" {
		t.Fatalf("index2 accumulation mismatch: %q", c.Choices[2].Message.Content.Text)
	}
}

func TestChatCompletionBuilder_BuildIdempotent(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "x"}}}})
	c1 := b.Build()
	c2 := b.Build()
	b2 := NewChatCompletionBuilder()
	b2.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{Content: "x"}}}})
	c3 := b2.Build()
	if c1.Choices[0].Message.Content.Text != c2.Choices[0].Message.Content.Text || c1.Choices[0].Message.Content.Text != c3.Choices[0].Message.Content.Text {
		t.Fatalf("build not idempotent")
	}
}

func TestFinishReason_MultipleChoicesFirstNonEmptyWins(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{
		{Index: 0, FinishReason: ChatCompletionFinishReasonLength},
		{Index: 1, FinishReason: ChatCompletionFinishReasonStop},
	}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{
		{Index: 0, FinishReason: ChatCompletionFinishReasonStop},
		{Index: 1, FinishReason: ChatCompletionFinishReasonLength},
	}})
	c := b.Build()
	if c.Choices[0].FinishReason != ChatCompletionFinishReasonLength {
		t.Fatalf("choice0 finish overwrote: %v", c.Choices[0].FinishReason)
	}
	if c.Choices[1].FinishReason != ChatCompletionFinishReasonStop {
		t.Fatalf("choice1 finish overwrote: %v", c.Choices[1].FinishReason)
	}
}

func TestUsageMerge_NextNilPreservePrevious(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}, Usage: &ChatCompletionUsage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7, Cost: json.Number("1.2")}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}})
	c := b.Build()
	if c.Usage == nil || c.Usage.PromptTokens != 3 || c.Usage.CompletionTokens != 4 || c.Usage.TotalTokens != 7 {
		t.Fatalf("usage not preserved: %+v", c.Usage)
	}
	if cf, _ := c.Usage.Cost.Float64(); math.Abs(cf-1.2) > 1e-12 {
		t.Fatalf("cost not preserved: %v", cf)
	}
}

func TestUsageMerge_NegativeAndZeroIgnored(t *testing.T) {
	b := NewChatCompletionBuilder()
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}, Usage: &ChatCompletionUsage{PromptTokens: 5, CompletionTokens: 6, TotalTokens: 11, Cost: json.Number("2.0")}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}, Usage: &ChatCompletionUsage{PromptTokens: 0, CompletionTokens: -1, TotalTokens: 0, Cost: json.Number("0")}})
	c := b.Build()
	if c.Usage.PromptTokens != 5 || c.Usage.CompletionTokens != 6 || c.Usage.TotalTokens != 11 {
		t.Fatalf("zero/negative overwrite occurred: %+v", c.Usage)
	}
	if cf, _ := c.Usage.Cost.Float64(); math.Abs(cf-2.0) > 1e-12 {
		t.Fatalf("cost overwrite occurred: %v", cf)
	}
}

func TestCostDetails_MergeAndOverrideRules(t *testing.T) {
	b := NewChatCompletionBuilder()
	x := json.Number("0")
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}, Usage: &ChatCompletionUsage{CostDetails: &ChatCompletionCostDetails{UpstreamInferenceCost: nil, UpstreamInferencePromptCost: json.Number("1.0"), UpstreamInferenceCompletionsCost: json.Number("2.0")}}})
	b.Add(&ChatCompletionChunk{Choices: []*ChatCompletionChunkChoice{{Index: 0, Delta: &ChatCompletionChunkChoiceDelta{}}}, Usage: &ChatCompletionUsage{CostDetails: &ChatCompletionCostDetails{UpstreamInferenceCost: &x, UpstreamInferencePromptCost: json.Number("1.5"), UpstreamInferenceCompletionsCost: json.Number("2.5")}}})
	c := b.Build()
	if c.Usage.CostDetails == nil || c.Usage.CostDetails.UpstreamInferenceCost == nil {
		t.Fatalf("upstream cost nil/zero merge failed: %+v", c.Usage.CostDetails)
	}
	if v, _ := c.Usage.CostDetails.UpstreamInferenceCost.Float64(); v != 0.0 {
		t.Fatalf("upstream cost not zero: %v", v)
	}
	if v1, _ := c.Usage.CostDetails.UpstreamInferencePromptCost.Float64(); v1 != 1.5 {
		t.Fatalf("prompt cost override failed: %v", v1)
	}
	if v2, _ := c.Usage.CostDetails.UpstreamInferenceCompletionsCost.Float64(); v2 != 2.5 {
		t.Fatalf("completions cost override failed: %v", v2)
	}
}

func TestReasoningAndText_AccumulationAndSwitch(t *testing.T) {
	mb := &ChatCompletionMessageBuilder{}
	mb.Add(&ChatCompletionChunkChoiceDelta{Content: "A", Reasoning: "r1"})
	mb.Add(&ChatCompletionChunkChoiceDelta{Content: "B", Reasoning: "r2"})
	m := mb.Build()
	if m.Content.Text != "AB" {
		t.Fatalf("text accumulation failed: %q", m.Content.Text)
	}
	if m.Reasoning != "r1r2" {
		t.Fatalf("reasoning accumulation failed: %q", m.Reasoning)
	}
}

func TestToolCalls_SkippedIndicesAndMultipleUpdates(t *testing.T) {
	mb := &ChatCompletionMessageBuilder{}
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 2, ID: "id2", Type: ChatCompletionMessageToolCallTypeFunction, Function: &ChatCompletionMessageToolCallFunction{Name: "f2", Arguments: "a"}}}})
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 0, ID: "id0", Type: ChatCompletionMessageToolCallTypeFunction, Function: &ChatCompletionMessageToolCallFunction{Name: "f0", Arguments: "x"}}}})
	mb.Add(&ChatCompletionChunkChoiceDelta{ToolCalls: []*ChatCompletionToolCall{{Index: 2, Function: &ChatCompletionMessageToolCallFunction{Arguments: "b"}}}})
	m := mb.Build()
	if len(m.ToolCalls) != 3 {
		t.Fatalf("toolcalls len: %d", len(m.ToolCalls))
	}
	if m.ToolCalls[2].Function.Name != "f2" || m.ToolCalls[2].Function.Arguments != "ab" {
		t.Fatalf("index2 merge failed: %+v", m.ToolCalls[2])
	}
	if m.ToolCalls[0].Function.Name != "f0" || m.ToolCalls[0].Function.Arguments != "x" {
		t.Fatalf("index0 not set: %+v", m.ToolCalls[0])
	}
}

func TestResponseFormat_RoundTripAndJSONSchema(t *testing.T) {
	f := ChatCompletionResponseFormat{Type: ChatCompletionResponseFormatTypeJSONObject}
	bts, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("marshal err: %v", err)
	}
	var f2 ChatCompletionResponseFormat
	if err := json.Unmarshal(bts, &f2); err != nil || f2.Type != ChatCompletionResponseFormatTypeJSONObject {
		t.Fatalf("roundtrip fail: %v %#v", err, f2)
	}
}

func TestStop_Unmarshal_Array(t *testing.T) {
	var s ChatCompletionStop
	if err := json.Unmarshal([]byte("[\"a\",\"b\"]"), &s); err != nil || len(s) != 2 || s[0] != "a" || s[1] != "b" {
		t.Fatalf("array stop fail: %v %#v", err, s)
	}
}

func TestWithProviderPreference_OverrideExisting(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}
	body := []byte(`{"messages":[],"model":"m","stream":true,"provider":{"only":["google-vertex"]}}`)
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	pref := &ProviderPreference{Only: []string{"anthropic"}}
	WithProviderPreference(pref)(req)
	b1, _ := io.ReadAll(req.Body)
	var r CreateChatCompletionRequest
	_ = json.Unmarshal(b1, &r)
	if r.Provider == nil || len(r.Provider.Only) != 1 || r.Provider.Only[0] != "anthropic" {
		t.Fatalf("provider not overridden: %#v", r.Provider)
	}
}

func TestWithIdentity_OverrideHeaders(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{"HTTP-Referer": []string{"old"}, "X-Title": []string{"old"}}
	WithIdentity("newr", "newt")(req)
	if req.Header.Get("HTTP-Referer") != "newr" || req.Header.Get("X-Title") != "newt" {
		t.Fatalf("identity headers not overridden")
	}
}

func TestErrorTypeMapping_Extended(t *testing.T) {
	codes := []int{401, 403, 408, 499, 600, 700}
	expects := []string{"invalid_request_error", "invalid_request_error", "invalid_request_error", "invalid_request_error", "unknown_error", "unknown_error"}
	for i, code := range codes {
		e := &Error{}
		e.Inner.Code = code
		if e.Type() != expects[i] {
			t.Fatalf("code %d type %s", code, e.Type())
		}
	}
}
