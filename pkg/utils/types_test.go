package utils

import (
	"encoding/json"
	"io"
	"testing"
)

func TestTrue_MarshalJSON(t *testing.T) {
	var v True = false
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(b) != "true" {
		t.Fatalf("expected true, got %s", string(b))
	}
}

func TestResettableReader_Reset(t *testing.T) {
	data := []byte("hello")
	r := NewResettableReader(data)
	rr, ok := r.(*ResettableReader)
	if !ok {
		t.Fatalf("expected *ResettableReader, got %T", r)
	}
	buf := make([]byte, 2)
	n, err := rr.Read(buf)
	if err != nil && err != io.EOF {
		t.Fatalf("unexpected read error: %v", err)
	}
	if n != 2 || string(buf) != "he" {
		t.Fatalf("expected he, got %q (n=%d)", string(buf), n)
	}
	if err := rr.Reset(); err != nil {
		t.Fatalf("unexpected reset error: %v", err)
	}
	all, err := io.ReadAll(rr)
	if err != nil {
		t.Fatalf("unexpected readall error: %v", err)
	}
	if string(all) != "hello" {
		t.Fatalf("expected hello after reset, got %q", string(all))
	}
}
