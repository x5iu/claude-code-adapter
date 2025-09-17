package provider

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/utils"
)

type ResponseHandler struct {
	CurrentMethod string
	Response      *http.Response
}

var providerErrorParser = map[string]func(*http.Response) error{
	ProviderMethodMakeAnthropicMessagesRequest:   parseError[*anthropic.Error],
	ProviderMethodGenerateAnthropicMessage:       parseError[*anthropic.Error],
	ProviderMethodCountAnthropicTokens:           parseError[*anthropic.Error],
	ProviderMethodCreateOpenRouterChatCompletion: parseError[*openrouter.Error],
}

func (r *ResponseHandler) ScanValues(values ...any) error {
	for _, dst := range values {
		if header, isHeader := dst.(*http.Header); isHeader {
			*header = r.Response.Header
		}
	}
	if r.Response.StatusCode/100 != 2 {
		return providerErrorParser[r.CurrentMethod](r.Response)
	}
	responseHeader := r.Response.Header
	switch r.CurrentMethod {
	case ProviderMethodMakeAnthropicMessagesRequest:
		readCloser := values[0].(*io.ReadCloser)
		*readCloser = r.Response.Body
		return nil
	case ProviderMethodGenerateAnthropicMessage:
		if !utils.IsContentType(responseHeader, "text/event-stream") {
			return fmt.Errorf("unexpected Content-Type: %s", responseHeader.Get("Content-Type"))
		}
		stream := values[0].(*anthropic.MessageStream)
		*stream = makeAnthropicStream(r.Response.Body)
	case ProviderMethodCreateOpenRouterChatCompletion:
		if !utils.IsContentType(responseHeader, "text/event-stream") {
			return fmt.Errorf("unexpected Content-Type: %s", responseHeader.Get("Content-Type"))
		}
		stream := values[0].(*openrouter.ChatCompletionStream)
		*stream = makeOpenRouterStream(r.Response.Body)
	default:
		defer r.Response.Body.Close()
		switch {
		case utils.IsContentType(responseHeader, "application/json"):
			jsonBody, err := io.ReadAll(r.Response.Body)
			if err != nil {
				return err
			}
			for _, dst := range values {
				if err = json.Unmarshal(jsonBody, dst); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unexpected Content-Type: %s", responseHeader.Get("Content-Type"))
		}
	}
	return nil
}

func (r *ResponseHandler) FromResponse(
	CurrentMethod string,
	response *http.Response,
) error {
	r.CurrentMethod = CurrentMethod
	r.Response = response
	return nil
}

func (ResponseHandler) Err() error  { return nil }
func (ResponseHandler) Break() bool { return true }

var anthropicEventBuilder = map[anthropic.EventType]func([]byte) (anthropic.Event, error){
	anthropic.EventTypePing:              unmarshalAnthropicEvent[*anthropic.EventPing],
	anthropic.EventTypeError:             unmarshalAnthropicEvent[*anthropic.EventError],
	anthropic.EventTypeMessageStart:      unmarshalAnthropicEvent[*anthropic.EventMessageStart],
	anthropic.EventTypeMessageDelta:      unmarshalAnthropicEvent[*anthropic.EventMessageDelta],
	anthropic.EventTypeMessageStop:       unmarshalAnthropicEvent[*anthropic.EventMessageStop],
	anthropic.EventTypeContentBlockStart: unmarshalAnthropicEvent[*anthropic.EventContentBlockStart],
	anthropic.EventTypeContentBlockDelta: unmarshalAnthropicEvent[*anthropic.EventContentBlockDelta],
	anthropic.EventTypeContentBlockStop:  unmarshalAnthropicEvent[*anthropic.EventContentBlockStop],
}

func unmarshalAnthropicEvent[E anthropic.Event](data []byte) (anthropic.Event, error) {
	var event E
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	return event, nil
}

func makeAnthropicStream(r io.ReadCloser) anthropic.MessageStream {
	return func(yield func(anthropic.Event, error) bool) {
		defer r.Close()
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			eventType, isEvent := bytes.CutPrefix(line, []byte("event:"))
			eventType = bytes.Clone(eventType) // next Scan overwrites bytes under eventType, a Clone keeps it unchanged
			if isEvent && scanner.Scan() {
				data, isData := bytes.CutPrefix(bytes.TrimSpace(scanner.Bytes()), []byte("data:"))
				if !isData {
					yield(nil, fmt.Errorf("missing anthropic %q data chunk", string(eventType)))
					return
				}
				if unmarshalEvent, ok := anthropicEventBuilder[anthropic.EventType(bytes.TrimSpace(eventType))]; ok {
					event, err := unmarshalEvent(data)
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(event, nil) {
						return
					}
				}
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
		}
	}
}

func makeDataIterator(r io.ReadCloser) iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		defer r.Close()
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			var isDataChunk bool
			line, isDataChunk = bytes.CutPrefix(line, []byte("data:"))
			if !isDataChunk {
				continue
			}
			line = bytes.TrimSpace(line)
			if bytes.EqualFold(line, []byte("[DONE]")) {
				return
			}
			var data json.RawMessage
			if err := json.Unmarshal(line, &data); err != nil {
				yield(nil, err)
				return
			}
			if !yield(data, nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
		}
	}
}

func makeOpenRouterStream(r io.ReadCloser) openrouter.ChatCompletionStream {
	dataIterator := makeDataIterator(r)
	return func(yield func(*openrouter.ChatCompletionChunk, error) bool) {
		for data, err := range dataIterator {
			if err != nil {
				yield(nil, err)
				return
			}
			var orError *openrouter.Error
			if err := json.Unmarshal(data, &orError); err == nil && orError.Inner.Code != 0 {
				yield(nil, orError)
				return
			}
			var chunk *openrouter.ChatCompletionChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				yield(nil, err)
				return
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

type Error interface {
	error

	Type() string
	Message() string
	Source() string

	StatusCode() int
	SetStatusCode(int)
}

func ParseError(err error) (e Error, is bool) {
	return e, errors.As(err, &e)
}

func parseError[E Error](r *http.Response) error {
	defer r.Body.Close()
	var e E
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	if utils.IsContentType(r.Header, "application/json") {
		if err = json.Unmarshal(body, &e); err != nil {
			return err
		}
		e.SetStatusCode(r.StatusCode)
		return e
	} else {
		return errors.New(string(body))
	}
}
