package utils

import (
	"net/http"
	"testing"
)

type encStruct struct {
	A int    `json:"a"`
	B string `json:"b"`
}

func TestJSONEncode_Success(t *testing.T) {
	s := encStruct{A: 2, B: "x"}
	got, err := JSONEncode(s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "{\"a\":2,\"b\":\"x\"}" {
		t.Fatalf("unexpected json: %s", got)
	}
}

func TestJSONEncode_Error(t *testing.T) {
	ch := make(chan int)
	_, err := JSONEncode(ch)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestJSONEncodeString_Success(t *testing.T) {
	s := encStruct{A: 1, B: "y"}
	got := JSONEncodeString(s)
	if got != "{\"a\":1,\"b\":\"y\"}" {
		t.Fatalf("unexpected json: %s", got)
	}
}

func TestJSONEncodeString_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic, got none")
		}
	}()
	_ = JSONEncodeString(func() {})
}

func TestIsContentType(t *testing.T) {
	h := http.Header{"Content-Type": []string{"application/json; charset=utf-8"}}
	if !IsContentType(h, "application/json") {
		t.Fatalf("expected true for application/json with charset")
	}
	if IsContentType(h, "text/plain") {
		t.Fatalf("expected false for text/plain")
	}
	h = http.Header{"Content-Type": []string{"application/json ; charset=utf-8"}}
	if !IsContentType(h, "application/json") {
		t.Fatalf("expected true for application/json with space before params")
	}
	h = http.Header{}
	if IsContentType(h, "application/json") {
		t.Fatalf("expected false when header missing")
	}
}

func TestStack_NonEmpty(t *testing.T) {
	b := stack(0)
	if len(b) == 0 {
		t.Fatalf("expected non-empty stack trace")
	}
}
