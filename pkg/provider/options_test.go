package provider

import (
	"bytes"
	"io"
	"net/http"
	"testing"
)

func TestWithHeaders(t *testing.T) {
	headers := http.Header{
		"X-Custom-Header": []string{"value1", "value2"},
		"Authorization":   []string{"Bearer token"},
	}

	req, err := http.NewRequest("GET", "https://example.com", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	option := WithHeaders(headers)
	option(req)

	// Check that headers were added
	if len(req.Header.Values("X-Custom-Header")) != 2 {
		t.Errorf("Expected 2 values for X-Custom-Header, got %d", len(req.Header.Values("X-Custom-Header")))
	}

	if req.Header.Get("Authorization") != "Bearer token" {
		t.Errorf("Expected Authorization header to be 'Bearer token', got '%s'", req.Header.Get("Authorization"))
	}

	expectedValues := []string{"value1", "value2"}
	actualValues := req.Header.Values("X-Custom-Header")
	for i, expected := range expectedValues {
		if i >= len(actualValues) || actualValues[i] != expected {
			t.Errorf("Expected X-Custom-Header[%d] to be '%s', got '%s'", i, expected, actualValues[i])
		}
	}
}

func TestReplaceBody(t *testing.T) {
	originalBody := "original body content"
	newBody := []byte("new body content")

	req, err := http.NewRequest("POST", "https://example.com", bytes.NewBufferString(originalBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	originalContentLength := req.ContentLength

	option := ReplaceBody(newBody)
	option(req)

	// Check that body was replaced
	if req.ContentLength != int64(len(newBody)) {
		t.Errorf("Expected ContentLength to be %d, got %d", len(newBody), req.ContentLength)
	}

	if req.ContentLength == originalContentLength {
		t.Error("ContentLength should have changed after replacing body")
	}

	// Read the body and verify content
	bodyContent, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("Failed to read request body: %v", err)
	}

	if string(bodyContent) != string(newBody) {
		t.Errorf("Expected body content to be '%s', got '%s'", string(newBody), string(bodyContent))
	}

	// Test GetBody function
	if req.GetBody == nil {
		t.Error("GetBody function should be set")
	} else {
		getBodyReader, err := req.GetBody()
		if err != nil {
			t.Fatalf("Failed to get body reader: %v", err)
		}
		getBodyContent, err := io.ReadAll(getBodyReader)
		if err != nil {
			t.Fatalf("Failed to read from GetBody: %v", err)
		}
		if string(getBodyContent) != string(newBody) {
			t.Errorf("Expected GetBody content to be '%s', got '%s'", string(newBody), string(getBodyContent))
		}
	}
}

func TestWithQuery(t *testing.T) {
	req, err := http.NewRequest("GET", "https://example.com?existing=value", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	option := WithQuery("new_param", "new_value")
	option(req)

	query := req.URL.Query()

	// Check that existing parameter is preserved
	if query.Get("existing") != "value" {
		t.Errorf("Expected existing parameter to be preserved, got '%s'", query.Get("existing"))
	}

	// Check that new parameter was added
	if query.Get("new_param") != "new_value" {
		t.Errorf("Expected new_param to be 'new_value', got '%s'", query.Get("new_param"))
	}
}

func TestMultipleOptions(t *testing.T) {
	headers := http.Header{
		"X-Test-Header": []string{"test-value"},
	}
	bodyData := []byte("test body")

	req, err := http.NewRequest("POST", "https://example.com", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	// Apply multiple options
	WithHeaders(headers)(req)
	ReplaceBody(bodyData)(req)
	WithQuery("param", "value")(req)

	// Verify all options were applied
	if req.Header.Get("X-Test-Header") != "test-value" {
		t.Error("Header option was not applied")
	}

	if req.ContentLength != int64(len(bodyData)) {
		t.Error("Body replacement option was not applied")
	}

	if req.URL.Query().Get("param") != "value" {
		t.Error("Query option was not applied")
	}
}
