package jsonl

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/x5iu/claude-code-adapter/pkg/snapshot"
)

// io.Writer -> io.WriteCloser adapter for tests
type nopWriteCloser struct{ io.Writer }
func (nopWriteCloser) Close() error { return nil }

func TestRecord_EnqueueAndCallbackSuccess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := &Recorder{cx: ctx, ch: make(chan *item, 1)}
	var got []byte
	done := make(chan struct{})
	go func() {
		it := <-r.ch
		got = it.snapshot
		it.report(ctx, nil)
		close(done)
	}()
	s := &snapshot.Snapshot{Version: "v1"}
	if err := r.Record(s); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	<-done
	want, _ := json.Marshal(s)
	if !bytes.Equal(got, want) {
		t.Fatalf("snapshot bytes mismatch")
	}
}

func TestRecord_EnqueueAndCallbackError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := &Recorder{cx: ctx, ch: make(chan *item, 1)}
	expected := errors.New("write failed")
	go func() {
		it := <-r.ch
		it.report(ctx, expected)
	}()
	if err := r.Record(&snapshot.Snapshot{Version: "v2"}); !errors.Is(err, expected) {
		t.Fatalf("expected error, got: %v", err)
	}
}

func TestNewRecorder_RecordWritesToFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "snap-*.jsonl")
	if err != nil {
		t.Fatalf("temp file error: %v", err)
	}
	defer f.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	rec := NewRecorder(ctx, f)
	s := &snapshot.Snapshot{Version: "x"}
	if err := rec.Record(s); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	b, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatalf("read file error: %v", err)
	}
	want, _ := json.Marshal(s)
	want = append(want, '\n')
	if !bytes.Equal(b, want) {
		t.Fatalf("file content mismatch: %q vs %q", string(b), string(want))
	}
}

func TestRecord_MultipleWritesNewlineSeparated(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "snap-*.jsonl")
	if err != nil {
		t.Fatalf("temp file error: %v", err)
	}
	defer f.Close()
	ctx := context.Background()
	rec := NewRecorder(ctx, f)
	s1 := &snapshot.Snapshot{Version: "a"}
	s2 := &snapshot.Snapshot{Version: "b"}
	if err := rec.Record(s1); err != nil {
		t.Fatal(err)
	}
	if err := rec.Record(s2); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	w1, _ := json.Marshal(s1)
	w2, _ := json.Marshal(s2)
	want := append(append(append([]byte{}, w1...), '\n'), append(w2, '\n')...)
	if !bytes.Equal(b, want) {
		t.Fatalf("multiple lines mismatch: %q vs %q", string(b), string(want))
	}
}

func TestClose_ReturnsAfterDraining(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "snap-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rec := NewRecorder(context.Background(), f)
	if err := rec.Record(&snapshot.Snapshot{Version: "z"}); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { _ = rec.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatalf("close did not return in time")
	}
}

func TestRecordAfterClose_ReturnsErrClosed(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "snap-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rec := NewRecorder(context.Background(), f)
	_ = rec.Close()
	if err := rec.Record(&snapshot.Snapshot{Version: "after"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}

func TestItemReport_Behavior(t *testing.T) {
	i := &item{callback: make(chan error, 1)}
	ctx := context.Background()
	expected := errors.New("x")
	go i.report(ctx, expected)
	select {
	case err := <-i.callback:
		if !errors.Is(err, expected) {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(50 * time.Millisecond):
		t.Fatalf("callback not received")
	}
}

func TestItemReport_DropsWhenContextDoneWithoutReceiver(t *testing.T) {
	i := &item{callback: make(chan error)}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan struct{})
	go func() {
		i.report(ctx, errors.New("x"))
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("report did not return after context done")
	}
	select {
	case <-i.callback:
		t.Fatalf("callback should be empty for unbuffered channel when context done")
	case <-time.After(50 * time.Millisecond):
	}
}

func TestPipeLargeWrite_IsNewlineTerminated(t *testing.T) {
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer pr.Close()
	defer pw.Close()
	ctx := context.Background()
	rec := NewRecorder(ctx, pw)
	defer rec.Close()
	lineCh := make(chan []byte, 1)
	errCh := make(chan error, 1)
	go func() {
		br := bufio.NewReader(pr)
		line, err := br.ReadBytes('\n')
		if err != nil {
			errCh <- err
			return
		}
		lineCh <- line
	}()
	big := make([]byte, 128*1024)
	for i := range big {
		big[i] = 'a'
	}
	s := &snapshot.Snapshot{Version: string(big)}
	if err := rec.Record(s); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-errCh:
		t.Fatal(err)
	case line := <-lineCh:
		if len(line) == 0 || line[len(line)-1] != '\n' {
			t.Fatalf("line not newline-terminated")
		}
	case <-time.After(time.Second):
		t.Fatalf("read timed out")
	}
}

type blockingWriter struct {
	wrote  chan struct{}
	resume chan struct{}
	once   sync.Once
	buf    bytes.Buffer
}

func (w *blockingWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.wrote) })
	<-w.resume
	return w.buf.Write(p)
}

func TestRecord_AfterEnqueue_CloseDoesNotAffectResult(t *testing.T) {
	w := &blockingWriter{wrote: make(chan struct{}), resume: make(chan struct{})}
	rec := NewRecorder(context.Background(), nopWriteCloser{w})
	errCh := make(chan error, 1)
	go func() { errCh <- rec.Record(&snapshot.Snapshot{Version: "x"}) }()
	<-w.wrote
	done := make(chan struct{})
	go func() { _ = rec.Close(); close(done) }()
	close(w.resume)
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatalf("timeout waiting for record")
	}
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("close did not return")
	}
}

func TestRecord_ContextCancelAfterEnqueue_ReturnsCtxErr(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	r := &Recorder{cx: ctx, ch: make(chan *item, 1)}
	received := make(chan struct{})
	go func() {
		it := <-r.ch
		close(received)
		time.Sleep(50 * time.Millisecond)
		it.report(ctx, nil)
	}()
	errCh := make(chan error, 1)
	go func() { errCh <- r.Record(&snapshot.Snapshot{Version: "c"}) }()
	<-received
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(time.Second):
		t.Fatalf("timeout waiting for error")
	}
}

func readLines(b *bytes.Buffer) []string {
	data := b.Bytes()
	lines := bytes.Split(data, []byte("\n"))
	res := make([]string, 0, len(lines))
	for _, ln := range lines {
		if len(ln) == 0 {
			continue
		}
		res = append(res, string(ln))
	}
	return res
}

func TestRecorder_ConcurrentRecord_NoLoss(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var buf bytes.Buffer
	r := NewRecorder(ctx, nopWriteCloser{&buf}).(*Recorder)
	r.flushEvery = 128
	N := 500
	wg := sync.WaitGroup{}
	errs := make([]error, N)
	for i := 0; i < N; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = r.Record(&snapshot.Snapshot{Version: fmt.Sprintf("v%03d", i)})
		}()
	}
	wg.Wait()
	if err := r.Close(); err != nil {
		t.Fatalf("close error: %v", err)
	}
	ok := 0
	for _, e := range errs {
		if e == nil {
			ok++
		}
	}
	lines := readLines(&buf)
	if len(lines) != ok {
		t.Fatalf("lines=%d ok=%d", len(lines), ok)
	}
	seen := map[string]bool{}
	for _, ln := range lines {
		var s snapshot.Snapshot
		if err := json.Unmarshal([]byte(ln), &s); err != nil {
			t.Fatalf("json error: %v", err)
		}
		seen[s.Version] = true
	}
	cnt := 0
	for i := 0; i < N; i++ {
		v := fmt.Sprintf("v%03d", i)
		if seen[v] {
			cnt++
		}
	}
	if cnt != ok {
		t.Fatalf("decoded=%d ok=%d", cnt, ok)
	}
}

func TestRecorder_DrainOnClose_Concurrent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var buf bytes.Buffer
	r := NewRecorder(ctx, nopWriteCloser{&buf}).(*Recorder)
	r.flushEvery = 256
	N := 200
	errs := make([]error, N)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		wg2 := sync.WaitGroup{}
		for i := 0; i < N; i++ {
			i := i
			wg2.Add(1)
			go func() {
				defer wg2.Done()
				errs[i] = r.Record(&snapshot.Snapshot{Version: fmt.Sprintf("x%03d", i)})
			}()
		}
		wg2.Wait()
	}()
	time.Sleep(10 * time.Millisecond)
	if err := r.Close(); err != nil {
		t.Fatalf("close error: %v", err)
	}
	wg.Wait()
	lines := readLines(&buf)
	succeeded := 0
	for _, e := range errs {
		if e == nil {
			succeeded++
		}
	}
	if len(lines) != succeeded {
		t.Fatalf("drain lines=%d succeeded=%d", len(lines), succeeded)
	}
	versions := make([]string, 0, len(lines))
	for _, ln := range lines {
		var s snapshot.Snapshot
		if err := json.Unmarshal([]byte(ln), &s); err != nil {
			t.Fatalf("json error: %v", err)
		}
		versions = append(versions, s.Version)
	}
	sort.Strings(versions)
	seen := 0
	for i := 0; i < N; i++ {
		v := fmt.Sprintf("x%03d", i)
		j := sort.SearchStrings(versions, v)
		if j < len(versions) && versions[j] == v {
			seen++
		}
	}
	if seen != succeeded {
		t.Fatalf("seen=%d succeeded=%d", seen, succeeded)
	}
}

type errorWriter struct{}

func (w *errorWriter) Write(p []byte) (int, error) { return 0, fmt.Errorf("x") }

func TestRecorder_ErrorPropagation_WriteFlush(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r := NewRecorder(ctx, nopWriteCloser{&errorWriter{}}).(*Recorder)
	r.flushEvery = 1
	err := r.Record(&snapshot.Snapshot{Version: "bad"})
	if err == nil {
		t.Fatalf("expected error")
	}
	_ = r.Close()
}
