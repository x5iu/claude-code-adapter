package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openai"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/profile"
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
	ProviderMethodCreateOpenAIModelResponse:      parseError[*openai.Error],
}

func (r *ResponseHandler) ScanValues(values ...any) error {
	ctx := r.Response.Request.Context()
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
		*stream = MakeAnthropicStream(profile.MustFromContext(ctx), r.Response.Body)
	case ProviderMethodCreateOpenRouterChatCompletion:
		if !utils.IsContentType(responseHeader, "text/event-stream") {
			return fmt.Errorf("unexpected Content-Type: %s", responseHeader.Get("Content-Type"))
		}
		stream := values[0].(*openrouter.ChatCompletionStream)
		*stream = makeOpenRouterStream(profile.MustFromContext(ctx), r.Response.Body)
	case ProviderMethodCreateOpenAIModelResponse:
		if !utils.IsContentType(responseHeader, "text/event-stream") {
			return fmt.Errorf("unexpected Content-Type: %s", responseHeader.Get("Content-Type"))
		}
		stream := values[0].(*openai.ResponseStream)
		*stream = makeOpenAIStream(r.Response.Body)
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

var openaiEventBuilder = map[openai.EventType]func([]byte) (openai.Event, error){
	openai.EventTypeResponseAudioDelta:                      unmarshalOpenAIEvent[*openai.ResponseAudioDeltaEvent],
	openai.EventTypeResponseAudioDone:                       unmarshalOpenAIEvent[*openai.ResponseAudioDoneEvent],
	openai.EventTypeResponseAudioTranscriptDelta:            unmarshalOpenAIEvent[*openai.ResponseAudioTranscriptDeltaEvent],
	openai.EventTypeResponseAudioTranscriptDone:             unmarshalOpenAIEvent[*openai.ResponseAudioTranscriptDoneEvent],
	openai.EventTypeResponseCodeInterpreterCallCodeDelta:    unmarshalOpenAIEvent[*openai.ResponseCodeInterpreterCallCodeDeltaEvent],
	openai.EventTypeResponseCodeInterpreterCallCodeDone:     unmarshalOpenAIEvent[*openai.ResponseCodeInterpreterCallCodeDoneEvent],
	openai.EventTypeResponseCodeInterpreterCallCompleted:    unmarshalOpenAIEvent[*openai.ResponseCodeInterpreterCallCompletedEvent],
	openai.EventTypeResponseCodeInterpreterCallInProgress:   unmarshalOpenAIEvent[*openai.ResponseCodeInterpreterCallInProgressEvent],
	openai.EventTypeResponseCodeInterpreterCallInterpreting: unmarshalOpenAIEvent[*openai.ResponseCodeInterpreterCallInterpretingEvent],
	openai.EventTypeResponseCompleted:                       unmarshalOpenAIEvent[*openai.ResponseCompletedEvent],
	openai.EventTypeResponseContentPartAdded:                unmarshalOpenAIEvent[*openai.ResponseContentPartAddedEvent],
	openai.EventTypeResponseContentPartDone:                 unmarshalOpenAIEvent[*openai.ResponseContentPartDoneEvent],
	openai.EventTypeResponseCreated:                         unmarshalOpenAIEvent[*openai.ResponseCreatedEvent],
	openai.EventTypeError:                                   unmarshalOpenAIEvent[*openai.ResponseErrorEvent],
	openai.EventTypeResponseFileSearchCallCompleted:         unmarshalOpenAIEvent[*openai.ResponseFileSearchCallCompletedEvent],
	openai.EventTypeResponseFileSearchCallInProgress:        unmarshalOpenAIEvent[*openai.ResponseFileSearchCallInProgressEvent],
	openai.EventTypeResponseFileSearchCallSearching:         unmarshalOpenAIEvent[*openai.ResponseFileSearchCallSearchingEvent],
	openai.EventTypeResponseFunctionCallArgumentsDelta:      unmarshalOpenAIEvent[*openai.ResponseFunctionCallArgumentsDeltaEvent],
	openai.EventTypeResponseFunctionCallArgumentsDone:       unmarshalOpenAIEvent[*openai.ResponseFunctionCallArgumentsDoneEvent],
	openai.EventTypeResponseInProgress:                      unmarshalOpenAIEvent[*openai.ResponseInProgressEvent],
	openai.EventTypeResponseFailed:                          unmarshalOpenAIEvent[*openai.ResponseFailedEvent],
	openai.EventTypeResponseIncomplete:                      unmarshalOpenAIEvent[*openai.ResponseIncompleteEvent],
	openai.EventTypeResponseOutputItemAdded:                 unmarshalOpenAIEvent[*openai.ResponseOutputItemAddedEvent],
	openai.EventTypeResponseOutputItemDone:                  unmarshalOpenAIEvent[*openai.ResponseOutputItemDoneEvent],
	openai.EventTypeResponseReasoningSummaryPartAdded:       unmarshalOpenAIEvent[*openai.ResponseReasoningSummaryPartAddedEvent],
	openai.EventTypeResponseReasoningSummaryPartDone:        unmarshalOpenAIEvent[*openai.ResponseReasoningSummaryPartDoneEvent],
	openai.EventTypeResponseReasoningSummaryTextDelta:       unmarshalOpenAIEvent[*openai.ResponseReasoningSummaryTextDeltaEvent],
	openai.EventTypeResponseReasoningSummaryTextDone:        unmarshalOpenAIEvent[*openai.ResponseReasoningSummaryTextDoneEvent],
	openai.EventTypeResponseReasoningTextDelta:              unmarshalOpenAIEvent[*openai.ResponseReasoningTextDeltaEvent],
	openai.EventTypeResponseReasoningTextDone:               unmarshalOpenAIEvent[*openai.ResponseReasoningTextDoneEvent],
	openai.EventTypeResponseRefusalDelta:                    unmarshalOpenAIEvent[*openai.ResponseRefusalDeltaEvent],
	openai.EventTypeResponseRefusalDone:                     unmarshalOpenAIEvent[*openai.ResponseRefusalDoneEvent],
	openai.EventTypeResponseOutputTextDelta:                 unmarshalOpenAIEvent[*openai.ResponseTextDeltaEvent],
	openai.EventTypeResponseOutputTextDone:                  unmarshalOpenAIEvent[*openai.ResponseTextDoneEvent],
	openai.EventTypeResponseWebSearchCallCompleted:          unmarshalOpenAIEvent[*openai.ResponseWebSearchCallCompletedEvent],
	openai.EventTypeResponseWebSearchCallInProgress:         unmarshalOpenAIEvent[*openai.ResponseWebSearchCallInProgressEvent],
	openai.EventTypeResponseWebSearchCallSearching:          unmarshalOpenAIEvent[*openai.ResponseWebSearchCallSearchingEvent],
	openai.EventTypeResponseImageGenCallCompleted:           unmarshalOpenAIEvent[*openai.ResponseImageGenCallCompletedEvent],
	openai.EventTypeResponseImageGenCallGenerating:          unmarshalOpenAIEvent[*openai.ResponseImageGenCallGeneratingEvent],
	openai.EventTypeResponseImageGenCallInProgress:          unmarshalOpenAIEvent[*openai.ResponseImageGenCallInProgressEvent],
	openai.EventTypeResponseImageGenCallPartialImage:        unmarshalOpenAIEvent[*openai.ResponseImageGenCallPartialImageEvent],
	openai.EventTypeResponseMcpCallArgumentsDelta:           unmarshalOpenAIEvent[*openai.ResponseMcpCallArgumentsDeltaEvent],
	openai.EventTypeResponseMcpCallArgumentsDone:            unmarshalOpenAIEvent[*openai.ResponseMcpCallArgumentsDoneEvent],
	openai.EventTypeResponseMcpCallCompleted:                unmarshalOpenAIEvent[*openai.ResponseMcpCallCompletedEvent],
	openai.EventTypeResponseMcpCallFailed:                   unmarshalOpenAIEvent[*openai.ResponseMcpCallFailedEvent],
	openai.EventTypeResponseMcpCallInProgress:               unmarshalOpenAIEvent[*openai.ResponseMcpCallInProgressEvent],
	openai.EventTypeResponseMcpListToolsInProgress:          unmarshalOpenAIEvent[*openai.ResponseMcpListToolsInProgressEvent],
	openai.EventTypeResponseMcpListToolsCompleted:           unmarshalOpenAIEvent[*openai.ResponseMcpListToolsCompletedEvent],
	openai.EventTypeResponseMcpListToolsFailed:              unmarshalOpenAIEvent[*openai.ResponseMcpListToolsFailedEvent],
	openai.EventTypeResponseOutputTextAnnotationAdded:       unmarshalOpenAIEvent[*openai.ResponseOutputTextAnnotationAddedEvent],
	openai.EventTypeResponseQueued:                          unmarshalOpenAIEvent[*openai.ResponseQueuedEvent],
	openai.EventTypeResponseCustomToolCallInputDelta:        unmarshalOpenAIEvent[*openai.ResponseCustomToolCallInputDeltaEvent],
	openai.EventTypeResponseCustomToolCallInputDone:         unmarshalOpenAIEvent[*openai.ResponseCustomToolCallInputDoneEvent],
}

func unmarshalOpenAIEvent[E openai.Event](data []byte) (openai.Event, error) {
	var event E
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	return event, nil
}

func unmarshalAnthropicEvent[E anthropic.Event](data []byte) (anthropic.Event, error) {
	var event E
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	return event, nil
}

func MakeAnthropicStream(prof *profile.Profile, r io.ReadCloser) anthropic.MessageStream {
	buffer := make([]byte, prof.Options.GetStreamDataBufferSize())
	return func(yieldimpl func(anthropic.Event, error) bool) {
		defer r.Close()
		var (
			mu   sync.Mutex
			stop atomic.Bool
		)
		defer stop.Store(true)
		yield := func(event anthropic.Event, err error) bool {
			if stop.Load() {
				return false
			}
			mu.Lock()
			defer mu.Unlock()
			if stop.Load() {
				return false
			}
			if !yieldimpl(event, err) {
				stop.Store(true)
				return false
			}
			return true
		}
		pingCtx, pingCancel := context.WithCancel(context.Background())
		defer pingCancel()
		go func() {
			ticker := time.NewTicker(3 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-pingCtx.Done():
					return
				case <-ticker.C:
					if !yield(&anthropic.EventPing{Type: anthropic.EventTypePing}, nil) {
						return
					}
				}
			}
		}()
		scanner := bufio.NewScanner(r)
		scanner.Buffer(buffer, cap(buffer))
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

func makeOpenAIStream(r io.ReadCloser) openai.ResponseStream {
	return func(yield func(openai.Event, error) bool) {
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
					yield(nil, fmt.Errorf("missing openai %q data chunk", string(eventType)))
					return
				}
				if unmarshalEvent, ok := openaiEventBuilder[openai.EventType(bytes.TrimSpace(eventType))]; ok {
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

func makeDataIterator(prof *profile.Profile, r io.ReadCloser) iter.Seq2[json.RawMessage, error] {
	buffer := make([]byte, prof.Options.GetStreamDataBufferSize())
	return func(yield func(json.RawMessage, error) bool) {
		defer r.Close()
		scanner := bufio.NewScanner(r)
		scanner.Buffer(buffer, cap(buffer))
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

func makeOpenRouterStream(prof *profile.Profile, r io.ReadCloser) openrouter.ChatCompletionStream {
	dataIterator := makeDataIterator(prof, r)
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
