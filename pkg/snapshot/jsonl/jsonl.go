package jsonl

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"

	"github.com/x5iu/claude-code-adapter/pkg/snapshot"
)

var ErrClosed = errors.New("jsonl recorder closed")

func NewRecorder(ctx context.Context, out io.WriteCloser) snapshot.Recorder {
	record := &Recorder{
		cx:         ctx,
		ch:         make(chan *item, 64),
		bw:         bufio.NewWriterSize(out, 64*1024),
		out:        out,
		closed:     make(chan struct{}),
		flushEvery: 32,
	}
	record.start()
	return record
}

type Recorder struct {
	cx         context.Context
	wg         sync.WaitGroup
	ch         chan *item
	bw         *bufio.Writer
	out        io.WriteCloser
	closed     chan struct{}
	once       sync.Once
	pending    int
	flushEvery int
}

func (r *Recorder) start() {
	r.wg.Add(1)
	appendToFile := func(it *item) {
		if _, err := r.bw.Write(it.snapshot); err != nil {
			it.report(r.cx, err)
			return
		}
		if err := r.bw.WriteByte('\n'); err != nil {
			it.report(r.cx, err)
			return
		}
		r.pending++
		if r.flushEvery > 0 && (r.pending >= r.flushEvery || len(r.ch) == 0) {
			if err := r.bw.Flush(); err != nil {
				it.report(r.cx, err)
				return
			}
			r.pending = 0
		}
		it.report(r.cx, nil)
	}
	go func() {
		defer r.wg.Done()
		for {
			select {
			case <-r.closed:
				for {
					select {
					case it := <-r.ch:
						if it != nil {
							appendToFile(it)
						}
					default:
						return
					}
				}
			case it := <-r.ch:
				if it != nil {
					appendToFile(it)
				}
			}
		}
	}()
}

func (r *Recorder) Record(snap *snapshot.Snapshot) error {
	select {
	case <-r.cx.Done():
		return r.cx.Err()
	case <-r.closed:
		return ErrClosed
	default:
	}
	bytes, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	it := &item{snapshot: bytes, callback: make(chan error, 1)}
	select {
	case <-r.cx.Done():
		return r.cx.Err()
	case <-r.closed:
		return ErrClosed
	case r.ch <- it:
	}
	select {
	case <-r.cx.Done():
		return r.cx.Err()
	case err := <-it.callback:
		return err
	}
}

func (r *Recorder) Close() error {
	r.once.Do(func() {
		close(r.closed)
	})
	r.wg.Wait()
	if r.bw != nil {
		if err := r.bw.Flush(); err != nil {
			return err
		}
	}
	return r.out.Close()
}

type item struct {
	snapshot []byte
	callback chan error
}

func (i *item) report(ctx context.Context, err error) {
	select {
	case <-ctx.Done():
	case i.callback <- err:
	}
}
