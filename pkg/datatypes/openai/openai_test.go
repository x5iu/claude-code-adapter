package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

func assertJSONEqual(t *testing.T, got []byte, want string) {
	t.Helper()
	var gotAny any
	if err := json.Unmarshal(got, &gotAny); err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	var wantAny any
	if err := json.Unmarshal([]byte(want), &wantAny); err != nil {
		t.Fatalf("unexpected want json error: %v", err)
	}
	if !reflect.DeepEqual(gotAny, wantAny) {
		t.Fatalf("json mismatch: %s vs %s", string(got), want)
	}
}

func TestResponseConversationUnmarshal(t *testing.T) {
	var conv ResponseConversation
	if err := json.Unmarshal([]byte(`"abc"`), &conv); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if conv.ID != "abc" {
		t.Fatalf("unexpected id: %s", conv.ID)
	}
	conv = ResponseConversation{}
	if err := json.Unmarshal([]byte(`{"id":"xyz"}`), &conv); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if conv.ID != "xyz" {
		t.Fatalf("unexpected id: %s", conv.ID)
	}
	if err := json.Unmarshal([]byte(`123`), &conv); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseInputParamUnmarshal(t *testing.T) {
	var param ResponseInputParam
	if err := json.Unmarshal([]byte(`"hello"`), &param); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(param) != 1 || param[0].Message == nil {
		t.Fatalf("unexpected param: %#v", param)
	}
	if param[0].Message.Content[0].Text.Text != "hello" {
		t.Fatalf("unexpected text: %s", param[0].Message.Content[0].Text.Text)
	}
	if err := json.Unmarshal([]byte(`123`), &param); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseInputItemParamMarshal(t *testing.T) {
	msg := newResponseMessage("hi")
	input := ResponseInputItemParam{Message: msg}
	got, err := json.Marshal(&input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("unexpected marshal result: %s", string(got))
	}
	empty := ResponseInputItemParam{}
	nullJSON, err := json.Marshal(&empty)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(nullJSON) != "null" {
		t.Fatalf("expected null, got %s", string(nullJSON))
	}
}

func TestResponseInputItemParamUnmarshal(t *testing.T) {
	var s ResponseInputItemParam
	if err := json.Unmarshal([]byte(`"hello"`), &s); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.Message == nil || s.Message.Content[0].Text.Text != "hello" {
		t.Fatalf("unexpected message: %#v", s.Message)
	}
	var call ResponseInputItemParam
	callJSON := `{"type":"function_call","id":"1","call_id":"c","name":"f","arguments":"{}","status":"completed"}`
	if err := json.Unmarshal([]byte(callJSON), &call); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if call.FunctionCall == nil || call.FunctionCall.Name != "f" {
		t.Fatalf("unexpected function call: %#v", call.FunctionCall)
	}
	if err := json.Unmarshal([]byte(`{"type":"unknown"}`), &call); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseMessageContentJSON(t *testing.T) {
	var content ResponseMessageContent
	if err := json.Unmarshal([]byte(`"hello"`), &content); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content.Text == nil || content.Text.Type != ResponseMessageContentTypeInputText {
		t.Fatalf("unexpected content: %#v", content)
	}
	imageJSON := `{"type":"input_image","image_url":"u","detail":"high"}`
	if err := json.Unmarshal([]byte(imageJSON), &content); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content.Image == nil || content.Image.ImageUrl != "u" {
		t.Fatalf("unexpected image: %#v", content.Image)
	}
	textContent := ResponseMessageContent{Text: &ResponseMessageContentText{Type: ResponseMessageContentTypeOutputText, Text: "x"}}
	got, err := json.Marshal(&textContent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, got, `{"type":"output_text","text":"x"}`)
	if err := json.Unmarshal([]byte(`{"type":"unknown"}`), &content); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseAnnotationJSON(t *testing.T) {
	var annotation ResponseAnnotation
	jsonData := `{"type":"file_citation","file_id":"f","filename":"name","index":2}`
	if err := json.Unmarshal([]byte(jsonData), &annotation); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if annotation.FileCitation == nil || annotation.FileCitation.Filename != "name" {
		t.Fatalf("unexpected citation: %#v", annotation.FileCitation)
	}
	got, err := json.Marshal(&annotation)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, got, jsonData)
	if err := json.Unmarshal([]byte(`123`), &annotation); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseFormatJSON(t *testing.T) {
	var format ResponseFormat
	textJSON := `{"type":"text"}`
	if err := json.Unmarshal([]byte(textJSON), &format); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if format.Text == nil || format.Text.Type != ResponseFormatTypeText {
		t.Fatalf("unexpected text format: %#v", format.Text)
	}
	format = ResponseFormat{}
	schemaJSON := `{"type":"json_schema","name":"s","description":"d","strict":true,"schema":{"a":1}}`
	if err := json.Unmarshal([]byte(schemaJSON), &format); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if format.JSONSchema == nil || format.JSONSchema.Name != "s" {
		t.Fatalf("unexpected schema: %#v", format.JSONSchema)
	}
	got, err := json.Marshal(&format)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, got, schemaJSON)
	if err := json.Unmarshal([]byte(`123`), &format); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseToolJSON(t *testing.T) {
	var tool ResponseToolParam
	functionJSON := `{"type":"function","name":"foo"}`
	if err := json.Unmarshal([]byte(functionJSON), &tool); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tool.Function == nil || tool.Function.Name != "foo" {
		t.Fatalf("unexpected function: %#v", tool.Function)
	}
	got, err := json.Marshal(&tool)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, got, functionJSON)
	if err := json.Unmarshal([]byte(`{"type":"unknown"}`), &tool); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseToolChoiceJSON(t *testing.T) {
	var choice ResponseToolChoice
	if err := json.Unmarshal([]byte(`"auto"`), &choice); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if choice.Option != ChatCompletionToolChoiceOptionAuto {
		t.Fatalf("unexpected option: %s", choice.Option)
	}
	choice = ResponseToolChoice{}
	allowedJSON := `{"type":"allowed_tools","mode":"required","tools":[{"name":"x"}]}`
	if err := json.Unmarshal([]byte(allowedJSON), &choice); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if choice.Allowed == nil || choice.Allowed.Mode != ChatCompletionToolChoiceOptionRequired {
		t.Fatalf("unexpected allowed: %#v", choice.Allowed)
	}
	got, err := json.Marshal(&choice)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, got, allowedJSON)
	if err := json.Unmarshal([]byte(`{"type":"unknown"}`), &choice); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseMCPToolParamAllowedToolsUnmarshal(t *testing.T) {
	var allowed ResponseMCPToolParamAllowedTools
	if err := json.Unmarshal([]byte(`["a","b"]`), &allowed); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(allowed.List, []string{"a", "b"}) {
		t.Fatalf("unexpected list: %#v", allowed.List)
	}
	objectJSON := `{"read_only":true,"tool_names":["x"]}`
	if err := json.Unmarshal([]byte(objectJSON), &allowed); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allowed.Filter == nil || !allowed.Filter.ReadOnly {
		t.Fatalf("unexpected filter: %#v", allowed.Filter)
	}
	if err := json.Unmarshal([]byte(`123`), &allowed); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseMCPToolParamRequireApprovalUnmarshal(t *testing.T) {
	var require ResponseMCPToolParamRequireApproval
	if err := json.Unmarshal([]byte(`"always"`), &require); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if require.Type != ResponseMCPToolParamRequireApprovalTypeAlways {
		t.Fatalf("unexpected type: %s", require.Type)
	}
	objectJSON := `{"always":{"read_only":true}}`
	if err := json.Unmarshal([]byte(objectJSON), &require); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if require.Filter == nil || require.Filter.Always == nil || !require.Filter.Always.ReadOnly {
		t.Fatalf("unexpected filter: %#v", require.Filter)
	}
	if err := json.Unmarshal([]byte(`123`), &require); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseCodeInterpreterToolParamUnmarshal(t *testing.T) {
	var param ResponseCodeInterpreterToolParam
	if err := json.Unmarshal([]byte(`"code_interpreter"`), &param); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if param.Type != ResponseToolCallTypeCodeInterpreter {
		t.Fatalf("unexpected type: %s", param.Type)
	}
	param = ResponseCodeInterpreterToolParam{}
	objectJSON := `{"Text":"run","Options":{"type":"auto","file_ids":["1"]}}`
	if err := json.Unmarshal([]byte(objectJSON), &param); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if param.Container == nil || param.Container.Text != "run" {
		t.Fatalf("unexpected container: %#v", param.Container)
	}
	if err := json.Unmarshal([]byte(`123`), &param); err == nil {
		t.Fatalf("expected error")
	}
}

func TestResponseJSONSchemaObjectJSON(t *testing.T) {
	original := []byte(`{"a":1}`)
	var obj ResponseJSONSchemaObject
	if err := json.Unmarshal(original, &obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual([]byte(obj), original) {
		t.Fatalf("unexpected data: %s", string(obj))
	}
	marshal, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertJSONEqual(t, marshal, `{"a":1}`)
}

func TestEvent_Unmarshal_WebSearchSearching(t *testing.T) {
	var e ResponseWebSearchCallSearchingEvent
	jsonData := `{"type":"response.web_search_call.searching","item_id":"i","output_index":1,"sequence_number":2}`
	if err := json.Unmarshal([]byte(jsonData), &e); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.ItemID != "i" || e.OutputIndex != 1 || e.SequenceNumber != 2 || e.EventType() != EventTypeResponseWebSearchCallSearching {
		t.Fatalf("unexpected event: %#v", e)
	}
}

func TestEvent_Unmarshal_ContentPartAdded(t *testing.T) {
	var e ResponseContentPartAddedEvent
	jsonData := `{"type":"response.content_part.added","content_index":0,"item_id":"it","output_index":0,"part":{"type":"output_text","text":"x"},"sequence_number":1}`
	if err := json.Unmarshal([]byte(jsonData), &e); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Part == nil || e.Part.Type != ResponseMessageContentTypeOutputText || e.Part.Text != "x" {
		t.Fatalf("unexpected part: %#v", e.Part)
	}
}

func TestEvent_Unmarshal_TextDelta(t *testing.T) {
	var e ResponseTextDeltaEvent
	jsonData := `{"type":"response.output_text.delta","content_index":0,"delta":"he","item_id":"it","logprobs":[],"output_index":0,"sequence_number":1}`
	if err := json.Unmarshal([]byte(jsonData), &e); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Delta != "he" || e.ContentIndex != 0 || e.EventType() != EventTypeResponseOutputTextDelta {
		t.Fatalf("unexpected event: %#v", e)
	}
}

func TestEvent_Unmarshal_OutputItemAdded_Message(t *testing.T) {
	var e ResponseOutputItemAddedEvent
	jsonData := `{"type":"response.output_item.added","item":{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"hello"}]},"output_index":0,"sequence_number":1}`
	if err := json.Unmarshal([]byte(jsonData), &e); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Item == nil || e.Item.Message == nil || e.Item.Message.Role != ResponseMessageRoleAssistant || e.Item.Message.Content[0].Text.Text != "hello" {
		t.Fatalf("unexpected item: %#v", e.Item)
	}
}

func TestResponseBuilder_BuildInitial(t *testing.T) {
	b := NewResponseBuilder()
	if b == nil || b.Build() == nil {
		t.Fatalf("nil builder or response")
	}
	if b.Build() != b.Response {
		t.Fatalf("build not returning internal response")
	}
}

func TestResponseBuilder_AddCompletedSetsResponse(t *testing.T) {
	b := NewResponseBuilder()
	initial := b.Build()
	r := &Response{ID: "r1"}
	b.Add(&ResponseCompletedEvent{Response: r, SequenceNumber: 1, Type: EventTypeResponseCompleted})
	got := b.Build()
	if got != r || got == initial {
		t.Fatalf("response not set to completed response")
	}
}

func TestResponseBuilder_AddErrorPanics(t *testing.T) {
	b := NewResponseBuilder()
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic")
		}
	}()
	b.Add(ResponseErrorEvent{Message: "x", SequenceNumber: 1, Type: EventTypeError})
}

func TestResponseBuilder_AddNonCompletedDoesNotReplace(t *testing.T) {
	b := NewResponseBuilder()
	orig := b.Build()
	r := &Response{ID: "x"}
	b.Add(&ResponseInProgressEvent{Response: r, SequenceNumber: 1, Type: EventTypeResponseInProgress})
	if b.Build() != orig {
		t.Fatalf("non-completed event should not replace response")
	}
}
