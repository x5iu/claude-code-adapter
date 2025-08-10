package utils

import (
	"bytes"
	"encoding/json"
	"io"
)

type True bool

func (True) MarshalJSON() ([]byte, error) {
	return json.Marshal(true)
}

func NewResettableReader(data []byte) io.Reader {
	return &ResettableReader{bytes.NewReader(data)}
}

type ResettableReader struct {
	*bytes.Reader
}

func (r *ResettableReader) Reset() error {
	_, err := r.Seek(0, io.SeekStart)
	return err
}
