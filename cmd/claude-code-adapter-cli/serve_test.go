package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/snapshot"
)

func TestRespondError(t *testing.T) {
	tests := []struct {
		name           string
		status         int
		message        string
		wantErrorType  string
		wantRetryAfter bool
	}{
		{
			name:          "bad request",
			status:        http.StatusBadRequest,
			message:       "invalid input",
			wantErrorType: anthropic.InvalidRequestError,
		},
		{
			name:          "unauthorized",
			status:        http.StatusUnauthorized,
			message:       "no api key",
			wantErrorType: anthropic.AuthenticationError,
		},
		{
			name:          "forbidden",
			status:        http.StatusForbidden,
			message:       "access denied",
			wantErrorType: anthropic.PermissionError,
		},
		{
			name:          "not found",
			status:        http.StatusNotFound,
			message:       "resource missing",
			wantErrorType: anthropic.NotFoundError,
		},
		{
			name:           "too many requests",
			status:         http.StatusTooManyRequests,
			message:        "rate limit exceeded",
			wantErrorType:  anthropic.RateLimitError,
			wantRetryAfter: true,
		},
		{
			name:           "internal server error",
			status:         http.StatusInternalServerError,
			message:        "something went wrong",
			wantErrorType:  anthropic.APIError,
			wantRetryAfter: true,
		},
		{
			name:           "overloaded",
			status:         529,
			message:        "server overloaded",
			wantErrorType:  anthropic.OverloadedError,
			wantRetryAfter: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			respondError(w, tt.status, tt.message)

			resp := w.Result()
			if resp.StatusCode != tt.status {
				t.Errorf("StatusCode = %d, want %d", resp.StatusCode, tt.status)
			}

			if resp.Header.Get("Content-Type") != "application/json" {
				t.Errorf("Content-Type = %q, want application/json", resp.Header.Get("Content-Type"))
			}

			if tt.wantRetryAfter {
				if resp.Header.Get("Retry-After") == "" {
					t.Error("Expected Retry-After header")
				}
				if resp.Header.Get("X-Should-Retry") != "true" {
					t.Error("Expected X-Should-Retry header to be true")
				}
			}

			var errResp anthropic.Error
			if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
				t.Fatalf("Failed to decode response body: %v", err)
			}

			if errResp.ContentType != anthropic.ErrorContentType {
				t.Errorf("Error.Type = %q, want %q", errResp.ContentType, anthropic.ErrorContentType)
			}

			if errResp.Inner.Type != tt.wantErrorType {
				t.Errorf("Inner.Type = %q, want %q", errResp.Inner.Type, tt.wantErrorType)
			}

			if errResp.Inner.Message != tt.message {
				t.Errorf("Inner.Message = %q, want %q", errResp.Inner.Message, tt.message)
			}
		})
	}
}

func TestMakeSnapshotRecorder(t *testing.T) {
	t.Run("empty config", func(t *testing.T) {
		recorder, err := makeSnapshotRecorder(context.Background(), "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if recorder == nil {
			t.Fatal("Expected recorder, got nil")
		}
		// NopRecorder check - hard to check type directly as it's internal or unexported,
		// but we can check if it works without error
		if err := recorder.Record(&snapshot.Snapshot{}); err != nil {
			t.Errorf("Record failed: %v", err)
		}
	})

	t.Run("jsonl config", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "test.jsonl")
		cfg := "jsonl:" + path

		recorder, err := makeSnapshotRecorder(context.Background(), cfg)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer recorder.Close()

		// Verify file was created
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("File %s was not created", path)
		}
	})

	t.Run("invalid scheme", func(t *testing.T) {
		_, err := makeSnapshotRecorder(context.Background(), "invalid:config")
		if err == nil {
			t.Fatal("Expected error for invalid scheme, got nil")
		}
		expectedErr := "unsupported snapshot recorder type \"invalid\""
		if err.Error() != expectedErr {
			t.Errorf("Error = %q, want %q", err.Error(), expectedErr)
		}
	})
}
