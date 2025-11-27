package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"iter"

	"github.com/x5iu/claude-code-adapter/pkg/utils"
)

type Error struct {
	Inner struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   any    `json:"param,omitempty"`
		Code    string `json:"code"`
	} `json:"error"`

	statusCode int
}

func (e *Error) Error() string                { return fmt.Sprintf("%s: %s", e.Type(), e.Message()) }
func (e *Error) Type() string                 { return e.Inner.Type }
func (e *Error) Message() string              { return e.Inner.Message }
func (e *Error) Source() string               { return "openai" }
func (e *Error) StatusCode() int              { return e.statusCode }
func (e *Error) SetStatusCode(statusCode int) { e.statusCode = statusCode }

// CreateModelResponseRequest follows OpenAI response API request format
// reference: https://platform.openai.com/docs/api-reference/responses/create
type CreateModelResponseRequest struct {
	Background         *bool                      `json:"background,omitempty"`
	Conversation       *ResponseConversation      `json:"conversation,omitempty"`
	Include            []ResponseIncludable       `json:"include,omitempty"`
	Input              ResponseInputParam         `json:"input,omitempty"`
	Instructions       string                     `json:"instructions,omitempty"`
	MaxOutputTokens    *int                       `json:"max_output_tokens,omitempty"`
	MaxToolCalls       *int                       `json:"max_tool_calls,omitempty"`
	Metadata           ResponseMetadata           `json:"metadata,omitempty"`
	Model              string                     `json:"model,omitempty"`
	ParallelToolCalls  *bool                      `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID string                     `json:"previous_response_id,omitempty"`
	Prompt             *ResponsePromptParam       `json:"prompt,omitempty"`
	PromptCacheKey     string                     `json:"prompt_cache_key,omitempty"`
	Reasoning          *ResponseReasoning         `json:"reasoning,omitempty"`
	SafetyIdentifier   string                     `json:"safety_identifier,omitempty"`
	ServiceTier        ResponseServiceTier        `json:"service_tier,omitempty"`
	Store              *bool                      `json:"store,omitempty"`
	Stream             utils.True                 `json:"stream"`
	StreamOptions      *ResponseStreamOptions     `json:"stream_options,omitempty"`
	Temperature        *float64                   `json:"temperature,omitempty"`
	Text               *ResponseTextConfigParam   `json:"text,omitempty"`
	ToolChoice         *ResponseToolChoice        `json:"tool_choice,omitempty"`
	Tools              []*ResponseToolParam       `json:"tools,omitempty"`
	TopLogprobs        *int                       `json:"top_logprobs,omitempty"`
	TopP               *float64                   `json:"top_p,omitempty"`
	Truncation         ResponseTruncationStrategy `json:"truncation,omitempty"`
	User               string                     `json:"user,omitempty"`
}

type ResponseConversation struct {
	ID string `json:"id"`
}

func (rc *ResponseConversation) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			var id string
			if err := json.Unmarshal(data, &id); err != nil {
				return err
			}
			rc.ID = id
			return nil
		case '{':
			var ir struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			rc.ID = ir.ID
			return nil
		default:
			return errors.New("conversation should be a string or an object")
		}
	}
	return errors.New("empty conversation")
}

type ResponseIncludable string

const (
	ResponseIncludableCodeInterpreterCallOutputs ResponseIncludable = "code_interpreter_call.outputs"
	ResponseIncludableComputerCallOutputImageUrl ResponseIncludable = "computer_call_output.output.image_url"
	ResponseIncludableFileSearchCallResults      ResponseIncludable = "file_search_call.results"
	ResponseIncludableMessageInputImageImageUrl  ResponseIncludable = "message.input_image.image_url"
	ResponseIncludableMessageOutputTextLogprobs  ResponseIncludable = "message.output_text.logprobs"
	ResponseIncludableReasoningEncryptedContent  ResponseIncludable = "reasoning.encrypted_content"
)

func TextInput(text string) ResponseInputParam {
	return ResponseInputParam{
		newResponseMessageInput(text),
	}
}

type ResponseInputParam []*ResponseInputItemParam

func (input *ResponseInputParam) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case 'n':
			return nil
		case '"':
			var text string
			if err := json.Unmarshal(data, &text); err != nil {
				return err
			}
			*input = append(*input, newResponseMessageInput(text))
			return nil
		case '[':
			return json.Unmarshal(data, &input)
		default:
			return errors.New("input should be a string or an array")
		}
	}
	return errors.New("empty input")
}

func newResponseMessageInput(text string) *ResponseInputItemParam {
	return &ResponseInputItemParam{
		Message: newResponseMessage(text),
	}
}

func newResponseMessage(text string) *ResponseMessage {
	return &ResponseMessage{
		Type: ResponseInputItemTypeMessage,
		Role: ResponseMessageRoleUser,
		Content: []*ResponseMessageContent{
			{
				Text: &ResponseMessageContentText{
					Type: ResponseMessageContentTypeInputText,
					Text: text,
				},
			},
		},
	}
}

// ResponseInputItemParam
// Text, image, or file inputs to the model, used to generate a response.
type ResponseInputItemParam struct {
	Message              *ResponseMessage
	FileSearchCall       *ResponseFileSearchToolCallParam
	ComputerCall         *ResponseComputerToolCallParam
	ComputerCallOutput   *ResponseComputerCallOutput
	WebSearchCall        *ResponseFunctionWebSearchParam
	FunctionCall         *ResponseFunctionToolCallParam
	FunctionCallOutput   *ResponseFunctionCallOutput
	Reasoning            *ResponseReasoningItem
	ImageGenerationCall  *ResponseImageGenerationCall
	CodeInterpreterCall  *ResponseCodeInterpreterToolCallParam
	LocalShellCall       *ResponseLocalShellCall
	LocalShellCallOutput *ResponseLocalShellCallOutput
	MCPListTools         *ResponseMCPListTools
	MCPApprovalRequest   *ResponseMCPApprovalRequest
	MCPApprovalResponse  *ResponseMCPApprovalResponse
	MCPCall              *ResponseMCPCall
	CustomToolCallOutput *ResponseCustomToolCallOutputParam
	CustomToolCall       *ResponseCustomToolCallParam
	ItemReference        *ResponseItemReference
}

func (input *ResponseInputItemParam) MarshalJSON() ([]byte, error) {
	if input.Message != nil {
		return json.Marshal(input.Message)
	}
	if input.FileSearchCall != nil {
		return json.Marshal(input.FileSearchCall)
	}
	if input.ComputerCall != nil {
		return json.Marshal(input.ComputerCall)
	}
	if input.ComputerCallOutput != nil {
		return json.Marshal(input.ComputerCallOutput)
	}
	if input.WebSearchCall != nil {
		return json.Marshal(input.WebSearchCall)
	}
	if input.FunctionCall != nil {
		return json.Marshal(input.FunctionCall)
	}
	if input.FunctionCallOutput != nil {
		return json.Marshal(input.FunctionCallOutput)
	}
	if input.Reasoning != nil {
		return json.Marshal(input.Reasoning)
	}
	if input.ImageGenerationCall != nil {
		return json.Marshal(input.ImageGenerationCall)
	}
	if input.CodeInterpreterCall != nil {
		return json.Marshal(input.CodeInterpreterCall)
	}
	if input.LocalShellCall != nil {
		return json.Marshal(input.LocalShellCall)
	}
	if input.LocalShellCallOutput != nil {
		return json.Marshal(input.LocalShellCallOutput)
	}
	if input.MCPListTools != nil {
		return json.Marshal(input.MCPListTools)
	}
	if input.MCPApprovalRequest != nil {
		return json.Marshal(input.MCPApprovalRequest)
	}
	if input.MCPApprovalResponse != nil {
		return json.Marshal(input.MCPApprovalResponse)
	}
	if input.MCPCall != nil {
		return json.Marshal(input.MCPCall)
	}
	if input.CustomToolCallOutput != nil {
		return json.Marshal(input.CustomToolCallOutput)
	}
	if input.CustomToolCall != nil {
		return json.Marshal(input.CustomToolCall)
	}
	if input.ItemReference != nil {
		return json.Marshal(input.ItemReference)
	}
	return json.Marshal(nil)
}

func (input *ResponseInputItemParam) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			var text string
			if err := json.Unmarshal(data, &text); err != nil {
				return err
			}
			input.Message = newResponseMessage(text)
			return nil
		case '{':
			var ir struct {
				Type ResponseInputItemType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseInputItemTypeMessage:
				return json.Unmarshal(data, &input.Message)
			case ResponseInputItemTypeFileSearchToolCall:
				return json.Unmarshal(data, &input.FileSearchCall)
			case ResponseInputItemTypeComputerToolCall:
				return json.Unmarshal(data, &input.ComputerCall)
			case ResponseInputItemTypeComputerCallOutput:
				return json.Unmarshal(data, &input.ComputerCallOutput)
			case ResponseInputItemTypeWebSearchCall:
				return json.Unmarshal(data, &input.WebSearchCall)
			case ResponseInputItemTypeFunctionCall:
				return json.Unmarshal(data, &input.FunctionCall)
			case ResponseInputItemTypeFunctionCallOutput:
				return json.Unmarshal(data, &input.FunctionCallOutput)
			case ResponseInputItemTypeReasoningItem:
				return json.Unmarshal(data, &input.Reasoning)
			case ResponseInputItemTypeImageGenerationCall:
				return json.Unmarshal(data, &input.ImageGenerationCall)
			case ResponseInputItemTypeCodeInterpreterCall:
				return json.Unmarshal(data, &input.CodeInterpreterCall)
			case ResponseInputItemTypeLocalShellCall:
				return json.Unmarshal(data, &input.LocalShellCall)
			case ResponseInputItemTypeLocalShellCallOutput:
				return json.Unmarshal(data, &input.LocalShellCallOutput)
			case ResponseInputItemTypeMCPListTools:
				return json.Unmarshal(data, &input.MCPListTools)
			case ResponseInputItemTypeMCPApprovalRequest:
				return json.Unmarshal(data, &input.MCPApprovalRequest)
			case ResponseInputItemTypeMCPApprovalResponse:
				return json.Unmarshal(data, &input.MCPApprovalResponse)
			case ResponseInputItemTypeMCPCall:
				return json.Unmarshal(data, &input.MCPCall)
			case ResponseInputItemTypeCustomToolCallOutput:
				return json.Unmarshal(data, &input.CustomToolCallOutput)
			case ResponseInputItemTypeCustomToolCall:
				return json.Unmarshal(data, &input.CustomToolCall)
			case ResponseInputItemTypeItemReference:
				return json.Unmarshal(data, &input.ItemReference)
			}
			return fmt.Errorf("unknown input item type %q", ir.Type)
		default:
			return errors.New("input item should be a string or an object")
		}
	}
	return errors.New("empty input item")
}

type ResponseInputItemType string

const (
	ResponseInputItemTypeMessage              ResponseInputItemType = "message"
	ResponseInputItemTypeFileSearchToolCall   ResponseInputItemType = "file_search_call"
	ResponseInputItemTypeComputerToolCall     ResponseInputItemType = "computer_call"
	ResponseInputItemTypeComputerCallOutput   ResponseInputItemType = "computer_call_output"
	ResponseInputItemTypeWebSearchCall        ResponseInputItemType = "web_search_call"
	ResponseInputItemTypeFunctionCall         ResponseInputItemType = "function_call"
	ResponseInputItemTypeFunctionCallOutput   ResponseInputItemType = "function_call_output"
	ResponseInputItemTypeReasoningItem        ResponseInputItemType = "reasoning"
	ResponseInputItemTypeImageGenerationCall  ResponseInputItemType = "image_generation_call"
	ResponseInputItemTypeCodeInterpreterCall  ResponseInputItemType = "code_interpreter_call"
	ResponseInputItemTypeLocalShellCall       ResponseInputItemType = "local_shell_call"
	ResponseInputItemTypeLocalShellCallOutput ResponseInputItemType = "local_shell_call_output"
	ResponseInputItemTypeMCPListTools         ResponseInputItemType = "mcp_list_tools"
	ResponseInputItemTypeMCPApprovalRequest   ResponseInputItemType = "mcp_approval_request"
	ResponseInputItemTypeMCPApprovalResponse  ResponseInputItemType = "mcp_approval_response"
	ResponseInputItemTypeMCPCall              ResponseInputItemType = "mcp_call"
	ResponseInputItemTypeCustomToolCallOutput ResponseInputItemType = "custom_tool_call_output"
	ResponseInputItemTypeCustomToolCall       ResponseInputItemType = "custom_tool_call"
	ResponseInputItemTypeItemReference        ResponseInputItemType = "item_reference"
)

type ResponseMessage struct {
	Type    ResponseInputItemType   `json:"type"`
	Role    ResponseMessageRole     `json:"role"`
	Content ResponseMessageContents `json:"content"`
	Status  ResponseMessageStatus   `json:"status"`
}

func NewTextContent(text string) ResponseMessageContents {
	return ResponseMessageContents{
		&ResponseMessageContent{
			Text: &ResponseMessageContentText{
				Type: ResponseMessageContentTypeInputText,
				Text: text,
			},
		},
	}
}

type ResponseMessageContents []*ResponseMessageContent

func (c *ResponseMessageContents) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			var text string
			if err := json.Unmarshal(data, &text); err != nil {
				return err
			}
			*c = NewTextContent(text)
			return nil
		case '[':
			// Use a type alias to avoid infinite recursion
			type rawContents []*ResponseMessageContent
			var contents rawContents
			if err := json.Unmarshal(data, &contents); err != nil {
				return err
			}
			*c = ResponseMessageContents(contents)
			return nil
		default:
			return errors.New("message content should be a string or an array")
		}
	}
	return errors.New("empty message content")
}

type ResponseMessageContent struct {
	Text  *ResponseMessageContentText
	Image *ResponseMessageContentImage
	File  *ResponseMessageContentFile
}

func (content *ResponseMessageContent) MarshalJSON() ([]byte, error) {
	if content.Text != nil {
		return json.Marshal(content.Text)
	}
	if content.Image != nil {
		return json.Marshal(content.Image)
	}
	if content.File != nil {
		return json.Marshal(content.File)
	}
	return json.Marshal(nil)
}

func (content *ResponseMessageContent) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			var text string
			if err := json.Unmarshal(data, &text); err != nil {
				return err
			}
			content.Text = &ResponseMessageContentText{
				Type: ResponseMessageContentTypeInputText,
				Text: text,
			}
			return nil
		case '{':
			var ir struct {
				Type ResponseMessageContentType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseMessageContentTypeInputText,
				ResponseMessageContentTypeOutputText, ResponseMessageContentTypeRefusal:
				return json.Unmarshal(data, &content.Text)
			case ResponseMessageContentTypeInputImage:
				return json.Unmarshal(data, &content.Image)
			case ResponseMessageContentTypeInputFile:
				return json.Unmarshal(data, &content.File)
			}
			return fmt.Errorf("unknown message content type %q", ir.Type)
		default:
			return errors.New("message content should be a string or an object")
		}
	}
	return errors.New("empty message content")
}

type ResponseMessageContentType string

const (
	ResponseMessageContentTypeInputText  ResponseMessageContentType = "input_text"
	ResponseMessageContentTypeInputImage ResponseMessageContentType = "input_image"
	ResponseMessageContentTypeInputFile  ResponseMessageContentType = "input_file"
	ResponseMessageContentTypeOutputText ResponseMessageContentType = "output_text"
	ResponseMessageContentTypeRefusal    ResponseMessageContentType = "refusal"
)

type ResponseMessageContentText struct {
	Type        ResponseMessageContentType `json:"type"`
	Text        string                     `json:"text"`
	Logprobs    []*ResponseLogprob         `json:"logprobs,omitempty"`
	Annotations []*ResponseAnnotation      `json:"annotations,omitempty"`
	Refusal     string                     `json:"refusal,omitempty"`
}

type ResponseLogprob struct {
	Token       string                       `json:"token"`
	Bytes       []int                        `json:"bytes"`
	Logprob     float64                      `json:"logprob"`
	TopLogprobs []*ResponseLogprobTopLogprob `json:"top_logprobs"`
}

type ResponseLogprobTopLogprob struct {
	Token   string  `json:"token"`
	Bytes   []int   `json:"bytes"`
	Logprob float64 `json:"logprob"`
}

type ResponseAnnotation struct {
	FileCitation          *ResponseAnnotationFileCitation
	URLCitation           *ResponseAnnotationURLCitation
	ContainerFileCitation *ResponseAnnotationContainerFileCitation
	FilePath              *ResponseAnnotationFilePath
}

func (annotation *ResponseAnnotation) MarshalJSON() ([]byte, error) {
	if annotation.FileCitation != nil {
		return json.Marshal(annotation.FileCitation)
	}
	if annotation.URLCitation != nil {
		return json.Marshal(annotation.URLCitation)
	}
	if annotation.ContainerFileCitation != nil {
		return json.Marshal(annotation.ContainerFileCitation)
	}
	if annotation.FilePath != nil {
		return json.Marshal(annotation.FilePath)
	}
	return json.Marshal(nil)
}

func (annotation *ResponseAnnotation) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '{':
			var ir struct {
				Type ResponseAnnotationType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseAnnotationTypeFileCitation:
				return json.Unmarshal(data, &annotation.FileCitation)
			case ResponseAnnotationTypeURLCitation:
				return json.Unmarshal(data, &annotation.URLCitation)
			case ResponseAnnotationTypeContainerFileCitation:
				return json.Unmarshal(data, &annotation.ContainerFileCitation)
			case ResponseAnnotationTypeFilePath:
				return json.Unmarshal(data, &annotation.FilePath)
			}
			return fmt.Errorf("unknown annotation type %q", ir.Type)
		default:
			return errors.New("annotation should be an object")
		}
	}
	return errors.New("empty annotation")
}

type ResponseAnnotationType string

const (
	ResponseAnnotationTypeFileCitation          ResponseAnnotationType = "file_citation"
	ResponseAnnotationTypeURLCitation           ResponseAnnotationType = "url_citation"
	ResponseAnnotationTypeContainerFileCitation ResponseAnnotationType = "container_file_citation"
	ResponseAnnotationTypeFilePath              ResponseAnnotationType = "file_path"
)

type ResponseAnnotationFileCitation struct {
	FileID   string                 `json:"file_id"`
	Filename string                 `json:"filename"`
	Index    int                    `json:"index"`
	Type     ResponseAnnotationType `json:"type"`
}

type ResponseAnnotationURLCitation struct {
	EndIndex   int                    `json:"end_index"`
	StartIndex int                    `json:"start_index"`
	Title      string                 `json:"title"`
	Type       ResponseAnnotationType `json:"type"`
	URL        string                 `json:"url"`
}

type ResponseAnnotationContainerFileCitation struct {
	ContainerID string                 `json:"container_id"`
	EndIndex    int                    `json:"end_index"`
	FileID      string                 `json:"file_id"`
	Filename    string                 `json:"filename"`
	StartIndex  int                    `json:"start_index"`
	Type        ResponseAnnotationType `json:"type"`
}

type ResponseAnnotationFilePath struct {
	FileID string                 `json:"file_id"`
	Index  int                    `json:"index"`
	Type   ResponseAnnotationType `json:"type"`
}

type ResponseMessageContentImage struct {
	Type     ResponseMessageContentType        `json:"type"`
	ImageUrl string                            `json:"image_url"`
	FileID   string                            `json:"file_id,omitempty"`
	Detail   ResponseMessageContentImageDetail `json:"detail,omitempty"`
}

type ResponseMessageContentImageDetail string

const (
	ResponseMessageContentImageDetailLow  ResponseMessageContentImageDetail = "low"
	ResponseMessageContentImageDetailHigh ResponseMessageContentImageDetail = "high"
	ResponseMessageContentImageDetailAuto ResponseMessageContentImageDetail = "auto"
)

type ResponseMessageContentFile struct {
	Type     ResponseMessageContentType `json:"type"`
	FileData string                     `json:"file_data"`
	FileID   string                     `json:"file_id,omitempty"`
	FileURL  string                     `json:"file_url"`
	Filename string                     `json:"filename"`
}

type ResponseMessageRole string

const (
	ResponseMessageRoleUser      ResponseMessageRole = "user"
	ResponseMessageRoleAssistant ResponseMessageRole = "assistant"
	ResponseMessageRoleSystem    ResponseMessageRole = "system"
	ResponseMessageRoleDeveloper ResponseMessageRole = "developer"
)

type ResponseMessageStatus string

const (
	ResponseMessageStatusInProgress ResponseMessageStatus = "in_progress"
	ResponseMessageStatusCompleted  ResponseMessageStatus = "completed"
	ResponseMessageStatusIncomplete ResponseMessageStatus = "incomplete"
)

type ResponseFileSearchToolCallParam struct {
	ID      string                      `json:"id"`
	Type    ResponseInputItemType       `json:"type"`
	Queries []string                    `json:"queries"`
	Status  ResponseStatus              `json:"status"`
	Results []*ResponseFileSearchResult `json:"results"`
}

type ResponseStatus string

const (
	ResponseStatusInProgress   ResponseStatus = "in_progress"
	ResponseStatusSearching    ResponseStatus = "searching"
	ResponseStatusGenerating   ResponseStatus = "generating"
	ResponseStatusInterpreting ResponseStatus = "interpreting"
	ResponseStatusCompleted    ResponseStatus = "completed"
	ResponseStatusIncomplete   ResponseStatus = "incomplete"
	ResponseStatusFailed       ResponseStatus = "failed"
)

type ResponseFileSearchResult struct {
	Attributes map[string]any `json:"attributes,omitempty"`
	FileID     string         `json:"file_id"`
	Filename   string         `json:"filename"`
	Score      float64        `json:"score"`
	Text       string         `json:"text"`
}

type ResponseComputerToolCallParam struct {
	ID                  string                               `json:"id"`
	Type                ResponseInputItemType                `json:"type"`
	Action              *ResponseComputerAction              `json:"action"`
	CallID              string                               `json:"call_id"`
	PendingSafetyChecks []*ResponseComputerActionSafetyCheck `json:"pending_safety_checks"`
	Status              ResponseStatus                       `json:"status"`
}

type ResponseComputerAction struct {
	Type    ResponseComputerActionType        `json:"type"`
	Button  string                            `json:"button,omitempty"`
	X       int                               `json:"x,omitempty"`
	Y       int                               `json:"y,omitempty"`
	Path    []*ResponseComputerActionDragPath `json:"path,omitempty"`
	Keys    []string                          `json:"keys,omitempty"`
	ScrollX int                               `json:"scroll_x,omitempty"`
	ScrollY int                               `json:"scroll_y,omitempty"`
	Text    string                            `json:"text,omitempty"`
}

type ResponseComputerActionType string

const (
	ResponseComputerActionTypeClick       ResponseComputerActionType = "click"
	ResponseComputerActionTypeDoubleClick ResponseComputerActionType = "double_click"
	ResponseComputerActionTypeDrag        ResponseComputerActionType = "drag"
	ResponseComputerActionTypeKeypress    ResponseComputerActionType = "keypress"
	ResponseComputerActionTypeMove        ResponseComputerActionType = "move"
	ResponseComputerActionTypeScreenshot  ResponseComputerActionType = "screenshot"
	ResponseComputerActionTypeScroll      ResponseComputerActionType = "scroll"
	ResponseComputerActionTypeType        ResponseComputerActionType = "type"
	ResponseComputerActionTypeWait        ResponseComputerActionType = "wait"
)

type ResponseComputerActionDragPath struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type ResponseComputerActionSafetyCheck struct {
	ID      string `json:"id"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

type ResponseComputerCallOutput struct {
	CallID                   string                               `json:"call_id"`
	Output                   *ResponseComputerActionScreenshot    `json:"output"`
	Type                     ResponseInputItemType                `json:"type"`
	ID                       string                               `json:"id"`
	AcknowledgedSafetyChecks []*ResponseComputerActionSafetyCheck `json:"acknowledged_safety_checks"`
	Status                   ResponseStatus                       `json:"status"`
}

type ResponseComputerActionScreenshot struct {
	Type     ResponseComputerActionScreenshotType `json:"type"`
	FileID   string                               `json:"file_id"`
	ImageURL string                               `json:"image_url"`
}

type ResponseComputerActionScreenshotType string

const (
	ResponseComputerActionScreenshotTypeComputerScreenshot ResponseComputerActionScreenshotType = "computer_screenshot"
)

type ResponseFunctionWebSearchParam struct {
	ID     string                           `json:"id"`
	Type   ResponseInputItemType            `json:"type"`
	Action *ResponseFunctionWebSearchAction `json:"action"`
	Status ResponseStatus                   `json:"status"`
}

type ResponseFunctionWebSearchAction struct {
	Type    ResponseFunctionWebSearchActionType `json:"type"`
	Query   string                              `json:"query,omitempty"`
	URL     string                              `json:"url,omitempty"`
	Pattern string                              `json:"pattern,omitempty"`
}

type ResponseFunctionWebSearchActionType string

const (
	ResponseFunctionWebSearchActionTypeSearch   ResponseFunctionWebSearchActionType = "search"
	ResponseFunctionWebSearchActionTypeOpenPage ResponseFunctionWebSearchActionType = "open_page"
	ResponseFunctionWebSearchActionTypeFind     ResponseFunctionWebSearchActionType = "find"
)

type ResponseFunctionToolCallParam struct {
	ID        string                `json:"id"`
	Type      ResponseInputItemType `json:"type"`
	Arguments string                `json:"arguments"`
	CallID    string                `json:"call_id"`
	Name      string                `json:"name"`
	Status    ResponseStatus        `json:"status"`
}

type ResponseFunctionCallOutput struct {
	CallID string                `json:"call_id"`
	Output string                `json:"output"`
	Type   ResponseInputItemType `json:"type"`
	ID     string                `json:"id"`
	Status ResponseStatus        `json:"status"`
}

type ResponseReasoningItem struct {
	ID               string                    `json:"id"`
	Type             ResponseInputItemType     `json:"type"`
	Summary          *ResponseReasoningContent `json:"summary"`
	Content          *ResponseReasoningContent `json:"content,omitempty"`
	Status           ResponseStatus            `json:"status"`
	EncryptedContent string                    `json:"encrypted_content,omitempty"`
}

type ResponseReasoningContent struct {
	Text string                `json:"text"`
	Type ResponseReasoningType `json:"type"`
}

type ResponseReasoningType string

const (
	ResponseReasoningTypeSummaryText   ResponseReasoningType = "summary_text"
	ResponseReasoningTypeReasoningText ResponseReasoningType = "reasoning_text"
)

type ResponseImageGenerationCall struct {
	ID     string                `json:"id"`
	Type   ResponseInputItemType `json:"type"`
	Result string                `json:"result"`
	Status ResponseStatus        `json:"status"`
}

type ResponseCodeInterpreterToolCallParam struct {
	ID          string                                   `json:"id"`
	Type        ResponseInputItemType                    `json:"type"`
	Code        string                                   `json:"code"`
	Status      ResponseStatus                           `json:"status"`
	ContainerID string                                   `json:"container_id"`
	Outputs     []*ResponseCodeInterpreterToolCallOutput `json:"outputs"`
}

type ResponseCodeInterpreterToolCallOutput struct {
	Type ResponseCodeInterpreterToolCallOutputType `json:"type"`
	Logs string                                    `json:"logs,omitempty"`
	URL  string                                    `json:"url,omitempty"`
}

type ResponseCodeInterpreterToolCallOutputType string

const (
	ResponseCodeInterpreterToolCallOutputTypeLogs  ResponseCodeInterpreterToolCallOutputType = "logs"
	ResponseCodeInterpreterToolCallOutputTypeImage ResponseCodeInterpreterToolCallOutputType = "image"
)

type ResponseLocalShellCall struct {
	ID     string                    `json:"id"`
	Type   ResponseInputItemType     `json:"type"`
	Action *ResponseLocalShellAction `json:"action"`
	CallID string                    `json:"call_id"`
	Status ResponseStatus            `json:"status,omitempty"`
}

type ResponseLocalShellAction struct {
	Type             ResponseLocalShellActionType `json:"type"`
	Command          []string                     `json:"command"`
	Env              map[string]string            `json:"env"`
	TimeoutMS        int                          `json:"timeout_ms,omitempty"`
	User             string                       `json:"user,omitempty"`
	WorkingDirectory string                       `json:"working_directory,omitempty"`
}

type ResponseLocalShellActionType string

const (
	ResponseLocalShellActionTypeExec ResponseLocalShellActionType = "exec"
)

type ResponseLocalShellCallOutput struct {
	ID     string                `json:"id"`
	Type   ResponseInputItemType `json:"type"`
	Output string                `json:"output"`
	Status ResponseStatus        `json:"status,omitempty"`
}

type ResponseMCPListTools struct {
	ID          string                      `json:"id"`
	ServerLabel string                      `json:"server_label"`
	Tools       []*ResponseMCPListToolsTool `json:"tools"`
	Type        ResponseInputItemType       `json:"type"`
	Error       string                      `json:"error,omitempty"`
}

type ResponseMCPListToolsTool struct {
	InputSchema ResponseJSONSchemaObject `json:"input_schema"`
	Name        string                   `json:"name"`
	Annotations ResponseJSONSchemaObject `json:"annotations,omitempty"`
	Description string                   `json:"description,omitempty"`
}

type ResponseMCPApprovalRequest struct {
	ID          string                `json:"id"`
	Arguments   string                `json:"arguments"`
	Name        string                `json:"name"`
	ServerLabel string                `json:"server_label"`
	Type        ResponseInputItemType `json:"type"`
}

type ResponseMCPApprovalResponse struct {
	ApprovalRequestID string                `json:"approval_request_id"`
	Approve           bool                  `json:"approve"`
	Type              ResponseInputItemType `json:"type"`
	ID                string                `json:"id,omitempty"`
	Reason            string                `json:"reason,omitempty"`
}

type ResponseMCPCall struct {
	ID          string                `json:"id"`
	Arguments   string                `json:"arguments"`
	Name        string                `json:"name"`
	ServerLabel string                `json:"server_label"`
	Type        ResponseInputItemType `json:"type"`
	Error       string                `json:"error,omitempty"`
	Output      string                `json:"output,omitempty"`
}

type ResponseCustomToolCallOutputParam struct {
	CallID string                `json:"call_id"`
	Output string                `json:"output"`
	Type   ResponseInputItemType `json:"type"`
	ID     string                `json:"id"`
}

type ResponseCustomToolCallParam struct {
	CallID string                `json:"call_id"`
	Input  string                `json:"input"`
	Name   string                `json:"name"`
	Type   ResponseInputItemType `json:"type"`
	ID     string                `json:"id"`
}

type ResponseItemReference struct {
	ID   string                `json:"id"`
	Type ResponseInputItemType `json:"type,omitempty"`
}

type ResponseMetadata map[string]string

// ResponsePromptParam
// reference: https://platform.openai.com/docs/guides/text?api-mode=responses#reusable-prompts
type ResponsePromptParam struct {
	ID        string                             `json:"id"`
	Variables map[string]*ResponseInputItemParam `json:"variables,omitempty"`
	Version   string                             `json:"version,omitempty"`
}

type ResponseReasoning struct {
	Effort          ResponseReasoningEffort  `json:"effort,omitempty"`
	GenerateSummary ResponseReasoningSummary `json:"generate_summary,omitempty"`
	Summary         ResponseReasoningSummary `json:"summary,omitempty"`
}

type ResponseReasoningEffort string

const (
	ResponseReasoningEffortMinimal ResponseReasoningEffort = "minimal"
	ResponseReasoningEffortLow     ResponseReasoningEffort = "low"
	ResponseReasoningEffortMedium  ResponseReasoningEffort = "medium"
	ResponseReasoningEffortHigh    ResponseReasoningEffort = "high"
)

type ResponseReasoningSummary string

const (
	ResponseReasoningSummaryAuto     ResponseReasoningSummary = "auto"
	ResponseReasoningSummaryConcise  ResponseReasoningSummary = "concise"
	ResponseReasoningSummaryDetailed ResponseReasoningSummary = "detailed"
)

type ResponseServiceTier string

const (
	ServiceTierAuto     ResponseServiceTier = "auto"
	ServiceTierDefault  ResponseServiceTier = "default"
	ServiceTierFlex     ResponseServiceTier = "flex"
	ServiceTierScale    ResponseServiceTier = "scale"
	ServiceTierPriority ResponseServiceTier = "priority"
)

type ResponseStreamOptions struct {
	IncludeObfuscation bool `json:"include_obfuscation"`
}

type ResponseJSONSchemaObject []byte

func (param ResponseJSONSchemaObject) MarshalJSON() ([]byte, error) {
	return []byte(param), nil
}

func (param *ResponseJSONSchemaObject) UnmarshalJSON(b []byte) error {
	*param = append((*param)[:0], b...)
	return nil
}

type ResponseFormat struct {
	Text       *ResponseFormatText
	JSONObject *ResponseFormatJSONObject
	JSONSchema *ResponseFormatJSONSchema
}

type ResponseFormatType string

const (
	ResponseFormatTypeText       ResponseFormatType = "text"
	ResponseFormatTypeJSONObject ResponseFormatType = "json_object"
	ResponseFormatTypeJSONSchema ResponseFormatType = "json_schema"
)

func (format ResponseFormat) MarshalJSON() ([]byte, error) {
	if format.Text != nil {
		return json.Marshal(format.Text)
	}
	if format.JSONObject != nil {
		return json.Marshal(format.JSONObject)
	}
	if format.JSONSchema != nil {
		return json.Marshal(format.JSONSchema)
	}
	return json.Marshal(nil)
}

func (format *ResponseFormat) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '{':
			var ir struct {
				Type ResponseFormatType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseFormatTypeText:
				return json.Unmarshal(data, &format.Text)
			case ResponseFormatTypeJSONObject:
				return json.Unmarshal(data, &format.JSONObject)
			case ResponseFormatTypeJSONSchema:
				return json.Unmarshal(data, &format.JSONSchema)
			default:
				return errors.New("invalid response_format, available types are 'text', 'json_object', 'json_schema'")
			}
		default:
			return errors.New("response_format should be an object")
		}
	}
	return errors.New("empty response_format")
}

type ResponseFormatText struct {
	Type ResponseFormatType `json:"type"`
}

type ResponseFormatJSONObject struct {
	Type ResponseFormatType `json:"type"`
}

type ResponseFormatJSONSchema struct {
	Type        ResponseFormatType       `json:"type"`
	Name        string                   `json:"name"`
	Description string                   `json:"description,omitempty"`
	Strict      bool                     `json:"strict,omitempty"`
	Schema      ResponseJSONSchemaObject `json:"schema,omitempty"`
}

type ResponseTextConfigParam struct {
	Format    *ResponseFormat   `json:"format"`
	Verbosity ResponseVerbosity `json:"verbosity,omitempty"`
}

type ResponseVerbosity string

const (
	ResponseVerbosityLow    ResponseVerbosity = "low"
	ResponseVerbosityMedium ResponseVerbosity = "medium"
	ResponseVerbosityHigh   ResponseVerbosity = "high"
)

type ResponseToolParam struct {
	Function        *ResponseFunctionToolParam
	FileSearch      *ResponseFileSearchToolParam
	WebSearch       *ResponseWebSearchToolParam
	ComputerUse     *ResponseComputerToolParam
	MCP             *ResponseMCPToolParam
	CodeInterpreter *ResponseCodeInterpreterToolParam
	ImageGeneration *ResponseImageGenerationToolParam
	LocalShell      *ResponseLocalShellToolParam
	Custom          *ResponseCustomToolParam
}

func (tool ResponseToolParam) MarshalJSON() ([]byte, error) {
	if tool.Function != nil {
		return json.Marshal(tool.Function)
	}
	if tool.FileSearch != nil {
		return json.Marshal(tool.FileSearch)
	}
	if tool.WebSearch != nil {
		return json.Marshal(tool.WebSearch)
	}
	if tool.ComputerUse != nil {
		return json.Marshal(tool.ComputerUse)
	}
	if tool.MCP != nil {
		return json.Marshal(tool.MCP)
	}
	if tool.CodeInterpreter != nil {
		return json.Marshal(tool.CodeInterpreter)
	}
	if tool.ImageGeneration != nil {
		return json.Marshal(tool.ImageGeneration)
	}
	if tool.LocalShell != nil {
		return json.Marshal(tool.LocalShell)
	}
	if tool.Custom != nil {
		return json.Marshal(tool.Custom)
	}
	return json.Marshal(nil)
}

func (tool *ResponseToolParam) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '{':
			var ir struct {
				Type ResponseToolCallType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseToolCallTypeFunction:
				return json.Unmarshal(data, &tool.Function)
			case ResponseToolCallTypeFileSearch:
				return json.Unmarshal(data, &tool.FileSearch)
			case ResponseToolCallTypeWebSearch:
				return json.Unmarshal(data, &tool.WebSearch)
			case ResponseToolCallTypeComputerUse:
				return json.Unmarshal(data, &tool.ComputerUse)
			case ResponseToolCallTypeMCP:
				return json.Unmarshal(data, &tool.MCP)
			case ResponseToolCallTypeCodeInterpreter:
				return json.Unmarshal(data, &tool.CodeInterpreter)
			case ResponseToolCallTypeImageGeneration:
				return json.Unmarshal(data, &tool.ImageGeneration)
			case ResponseToolCallTypeLocalShell:
				return json.Unmarshal(data, &tool.LocalShell)
			case ResponseToolCallTypeCustom:
				return json.Unmarshal(data, &tool.Custom)
			}
			return fmt.Errorf("unknown tool type %q", ir.Type)
		default:
			return errors.New("tool should be an object")
		}
	}
	return errors.New("empty tool")
}

type ResponseToolCallType string

const (
	ResponseToolCallTypeFunction          ResponseToolCallType = "function"
	ResponseToolCallTypeFileSearch        ResponseToolCallType = "file_search"
	ResponseToolCallTypeWebSearch         ResponseToolCallType = "web_search_preview"
	ResponseToolCallTypeWebSearch20250311 ResponseToolCallType = "web_search_preview_2025_03_11"
	ResponseToolCallTypeComputerUse       ResponseToolCallType = "computer_use_preview"
	ResponseToolCallTypeMCP               ResponseToolCallType = "mcp"
	ResponseToolCallTypeCodeInterpreter   ResponseToolCallType = "code_interpreter"
	ResponseToolCallTypeImageGeneration   ResponseToolCallType = "image_generation"
	ResponseToolCallTypeLocalShell        ResponseToolCallType = "local_shell"
	ResponseToolCallTypeCustom            ResponseToolCallType = "custom"
)

type ResponseFunctionToolParam struct {
	Type        ResponseToolCallType     `json:"type"`
	Name        string                   `json:"name"`
	Description string                   `json:"description,omitempty"`
	Strict      bool                     `json:"strict,omitempty"`
	Parameters  ResponseJSONSchemaObject `json:"parameters,omitempty"`
}

type ResponseFileSearchToolParam struct {
	Type           ResponseToolCallType                       `json:"type"`
	VectorStoreIDs []string                                   `json:"vector_store_ids"`
	Filters        *ResponseFileSearchToolParamFilters        `json:"filters,omitempty"`
	MaxNumResults  int                                        `json:"max_num_results"`
	RankingOptions *ResponseFileSearchToolParamRankingOptions `json:"ranking_options,omitempty"`
}

type ResponseFileSearchToolParamFilters struct {
	Type    ResponseFileSearchToolParamFiltersType `json:"type"`
	Key     string                                 `json:"key"`
	Value   any                                    `json:"value"`
	Filters []*ResponseFileSearchToolParamFilters  `json:"filters,omitempty"`
}

type ResponseFileSearchToolParamFiltersType string

// Comparison filters
const (
	ResponseFileSearchToolParamFiltersTypeEq  ResponseFileSearchToolParamFiltersType = "eq"
	ResponseFileSearchToolParamFiltersTypeNe  ResponseFileSearchToolParamFiltersType = "ne"
	ResponseFileSearchToolParamFiltersTypeGt  ResponseFileSearchToolParamFiltersType = "gt"
	ResponseFileSearchToolParamFiltersTypeGte ResponseFileSearchToolParamFiltersType = "gte"
	ResponseFileSearchToolParamFiltersTypeLt  ResponseFileSearchToolParamFiltersType = "lt"
	ResponseFileSearchToolParamFiltersTypeLte ResponseFileSearchToolParamFiltersType = "lte"
)

// Compound filters
const (
	ResponseFileSearchToolParamFiltersTypeAnd ResponseFileSearchToolParamFiltersType = "and"
	ResponseFileSearchToolParamFiltersTypeOr  ResponseFileSearchToolParamFiltersType = "or"
)

type ResponseFileSearchToolParamRankingOptions struct {
	Ranker         ResponseFileSearchToolParamRanker `json:"ranker"`
	ScoreThreshold float64                           `json:"score_threshold"`
}

type ResponseFileSearchToolParamRanker string

const (
	ResponseFileSearchToolParamRankerAuto            ResponseFileSearchToolParamRanker = "auto"
	ResponseFileSearchToolParamRankerDefault20241115 ResponseFileSearchToolParamRanker = "default-2024-11-15"
)

type ResponseWebSearchToolParam struct {
	Type              ResponseToolCallType                        `json:"type"`
	SearchContextSize ResponseWebSearchToolParamSearchContextSize `json:"search_context_size"`
	UserLocation      *ResponseWebSearchToolParamUserLocation     `json:"user_location,omitempty"`
}

type ResponseWebSearchToolParamSearchContextSize string

const (
	ResponseWebSearchToolParamSearchContextSizeLow    ResponseWebSearchToolParamSearchContextSize = "low"
	ResponseWebSearchToolParamSearchContextSizeMedium ResponseWebSearchToolParamSearchContextSize = "medium"
	ResponseWebSearchToolParamSearchContextSizeHigh   ResponseWebSearchToolParamSearchContextSize = "high"
)

type ResponseWebSearchToolParamUserLocation struct {
	Type     ResponseWebSearchToolParamUserLocationType `json:"type"`
	City     string                                     `json:"city,omitempty"`
	Country  string                                     `json:"country,omitempty"`
	Region   string                                     `json:"region,omitempty"`
	Timezone string                                     `json:"timezone,omitempty"`
}

type ResponseWebSearchToolParamUserLocationType string

const (
	ResponseWebSearchToolParamUserLocationTypeApproximate ResponseWebSearchToolParamUserLocationType = "approximate"
)

type ResponseComputerToolParam struct {
	Type          ResponseToolCallType                 `json:"type"`
	DisplayHeight int                                  `json:"display_height"`
	DisplayWidth  int                                  `json:"display_width"`
	Environment   ResponseComputerToolParamEnvironment `json:"environment"`
}

type ResponseComputerToolParamEnvironment string

const (
	ResponseComputerToolParamEnvironmentWindows ResponseComputerToolParamEnvironment = "windows"
	ResponseComputerToolParamEnvironmentMac     ResponseComputerToolParamEnvironment = "mac"
	ResponseComputerToolParamEnvironmentLinux   ResponseComputerToolParamEnvironment = "linux"
	ResponseComputerToolParamEnvironmentUbuntu  ResponseComputerToolParamEnvironment = "ubuntu"
	ResponseComputerToolParamEnvironmentBrowser ResponseComputerToolParamEnvironment = "browser"
)

type ResponseMCPToolParam struct {
	Type              ResponseToolCallType                 `json:"type"`
	ServerLabel       string                               `json:"server_label"`
	AllowedTools      *ResponseMCPToolParamAllowedTools    `json:"allowed_tools,omitempty"`
	Authorization     string                               `json:"authorization"`
	ConnectorID       ResponseMCPToolParamConnectorID      `json:"connector_id,omitempty"`
	Headers           map[string]string                    `json:"headers,omitempty"`
	RequireApproval   *ResponseMCPToolParamRequireApproval `json:"require_approval,omitempty"`
	ServerDescription string                               `json:"server_description,omitempty"`
	ServerUrl         string                               `json:"server_url,omitempty"`
}

type ResponseMCPToolParamAllowedTools struct {
	List   []string
	Filter *ResponseMCPToolFilter
}

func (param *ResponseMCPToolParamAllowedTools) MarshalJSON() ([]byte, error) {
	if param.List != nil {
		return json.Marshal(param.List)
	}
	if param.Filter != nil {
		return json.Marshal(param.Filter)
	}
	return json.Marshal(nil)
}

func (param *ResponseMCPToolParamAllowedTools) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '[':
			return json.Unmarshal(data, &param.List)
		case '{':
			return json.Unmarshal(data, &param.Filter)
		default:
			return errors.New("allowed_tools should be an array or an object")
		}
	}
	return errors.New("empty allowed_tools")
}

type ResponseMCPToolParamConnectorID string

const (
	ResponseMCPToolParamConnectorIDDropbox         ResponseMCPToolParamConnectorID = "connector_dropbox"
	ResponseMCPToolParamConnectorIDGmail           ResponseMCPToolParamConnectorID = "connector_gmail"
	ResponseMCPToolParamConnectorIDGoogleCalendar  ResponseMCPToolParamConnectorID = "connector_googlecalendar"
	ResponseMCPToolParamConnectorIDGoogleDrive     ResponseMCPToolParamConnectorID = "connector_googledrive"
	ResponseMCPToolParamConnectorIDMicrosoftTeams  ResponseMCPToolParamConnectorID = "connector_microsoftteams"
	ResponseMCPToolParamConnectorIDOutlookCalendar ResponseMCPToolParamConnectorID = "connector_outlookcalendar"
	ResponseMCPToolParamConnectorIDOutlookEmail    ResponseMCPToolParamConnectorID = "connector_outlookemail"
	ResponseMCPToolParamConnectorIDSharepoint      ResponseMCPToolParamConnectorID = "connector_sharepoint"
)

type ResponseMCPToolParamRequireApproval struct {
	Type   ResponseMCPToolParamRequireApprovalType
	Filter *ResponseMCPToolParamRequireApprovalFilter
}

func (param *ResponseMCPToolParamRequireApproval) MarshalJSON() ([]byte, error) {
	if param.Type != "" {
		return json.Marshal(param.Type)
	}
	if param.Filter != nil {
		return json.Marshal(param.Filter)
	}
	return json.Marshal(nil)
}

func (param *ResponseMCPToolParamRequireApproval) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			return json.Unmarshal(data, &param.Type)
		case '{':
			return json.Unmarshal(data, &param.Filter)
		default:
			return errors.New("require_approval should be a string or an object")
		}
	}
	return errors.New("empty require_approval")
}

type ResponseMCPToolParamRequireApprovalType string

const (
	ResponseMCPToolParamRequireApprovalTypeAlways ResponseMCPToolParamRequireApprovalType = "always"
	ResponseMCPToolParamRequireApprovalTypeNever  ResponseMCPToolParamRequireApprovalType = "never"
)

type ResponseMCPToolParamRequireApprovalFilter struct {
	Always *ResponseMCPToolFilter `json:"always,omitempty"`
	Never  *ResponseMCPToolFilter `json:"never,omitempty"`
}

type ResponseMCPToolFilter struct {
	ReadOnly  bool     `json:"read_only,omitempty"`
	ToolNames []string `json:"tool_names,omitempty"`
}

type ResponseCodeInterpreterToolParam struct {
	Type      ResponseToolCallType                       `json:"type"`
	Container *ResponseCodeInterpreterToolParamContainer `json:"container"`
}

type ResponseCodeInterpreterToolParamContainer struct {
	Text    string
	Options *ResponseCodeInterpreterContainerOptions
}

func (param *ResponseCodeInterpreterToolParam) MarshalJSON() ([]byte, error) {
	if param.Type != "" {
		return json.Marshal(param.Type)
	}
	if param.Container != nil {
		return json.Marshal(param.Container)
	}
	return json.Marshal(nil)
}

func (param *ResponseCodeInterpreterToolParam) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case '"':
			return json.Unmarshal(data, &param.Type)
		case '{':
			return json.Unmarshal(data, &param.Container)
		default:
			return errors.New("code_interpreter should be a string or an object")
		}
	}
	return errors.New("empty code_interpreter")
}

type ResponseCodeInterpreterContainerOptions struct {
	Type    ResponseCodeInterpreterContainerOptionsType `json:"type"`
	FileIDs []string                                    `json:"file_ids,omitempty"`
}

type ResponseCodeInterpreterContainerOptionsType string

const (
	ResponseCodeInterpreterContainerOptionsTypeAuto ResponseCodeInterpreterContainerOptionsType = "auto"
)

type ResponseImageGenerationToolParam struct {
	Type              ResponseToolCallType                   `json:"type"`
	Background        ResponseImageGenerationBackground      `json:"background,omitempty"`
	InputFidelity     ResponseImageGenerationInputFidelity   `json:"input_fidelity,omitempty"`
	InputImageMask    *ResponseImageGenerationInputImageMask `json:"input_image_mask,omitempty"`
	Model             ResponseImageGenerationModel           `json:"model,omitempty"`
	Moderation        ResponseImageGenerationModeration      `json:"moderation,omitempty"`
	OutputCompression *int                                   `json:"output_compression,omitempty"`
	OutputFormat      ResponseImageGenerationOutputFormat    `json:"output_format,omitempty"`
	PartialImages     *int                                   `json:"partial_images,omitempty"`
	Quality           ResponseImageGenerationQuality         `json:"quality,omitempty"`
	Size              ResponseImageGenerationSize            `json:"size,omitempty"`
}

type ResponseImageGenerationBackground string

const (
	ResponseImageGenerationBackgroundTransparent ResponseImageGenerationBackground = "transparent"
	ResponseImageGenerationBackgroundOpaque      ResponseImageGenerationBackground = "opaque"
	ResponseImageGenerationBackgroundAuto        ResponseImageGenerationBackground = "auto"
)

type ResponseImageGenerationInputFidelity string

const (
	ResponseImageGenerationInputFidelityHigh ResponseImageGenerationInputFidelity = "high"
	ResponseImageGenerationInputFidelityLow  ResponseImageGenerationInputFidelity = "low"
)

type ResponseImageGenerationInputImageMask struct {
	FileID   string `json:"file_id,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type ResponseImageGenerationModel string

const (
	ResponseImageGenerationModelGPTImage1 ResponseImageGenerationModel = "gpt-image-1"
)

type ResponseImageGenerationModeration string

const (
	ResponseImageGenerationModerationAuto ResponseImageGenerationModeration = "auto"
	ResponseImageGenerationModerationLow  ResponseImageGenerationModeration = "low"
)

type ResponseImageGenerationOutputFormat string

const (
	ResponseImageGenerationOutputFormatPNG  ResponseImageGenerationOutputFormat = "png"
	ResponseImageGenerationOutputFormatWebP ResponseImageGenerationOutputFormat = "webp"
	ResponseImageGenerationOutputFormatJPEG ResponseImageGenerationOutputFormat = "jpeg"
)

type ResponseImageGenerationQuality string

const (
	ResponseImageGenerationQualityLow    ResponseImageGenerationQuality = "low"
	ResponseImageGenerationQualityMedium ResponseImageGenerationQuality = "medium"
	ResponseImageGenerationQualityHigh   ResponseImageGenerationQuality = "high"
	ResponseImageGenerationQualityAuto   ResponseImageGenerationQuality = "auto"
)

type ResponseImageGenerationSize string

const (
	ResponseImageGenerationSize1024x1024 ResponseImageGenerationSize = "1024x1024"
	ResponseImageGenerationSize1024x1536 ResponseImageGenerationSize = "1024x1536"
	ResponseImageGenerationSize1536x1024 ResponseImageGenerationSize = "1536x1024"
	ResponseImageGenerationSizeAuto      ResponseImageGenerationSize = "auto"
)

type ResponseLocalShellToolParam struct {
	Type ResponseToolCallType `json:"type"`
}

type ResponseCustomToolParam struct {
	Type        ResponseToolCallType           `json:"type"`
	Name        string                         `json:"name"`
	Description string                         `json:"description,omitempty"`
	Format      *ResponseCustomToolParamFormat `json:"format,omitempty"`
}

type ResponseCustomToolParamFormat struct {
	Type       ResponseCustomToolFormatType   `json:"type"`
	Definition string                         `json:"definition,omitempty"`
	Syntax     ResponseCustomToolFormatSyntax `json:"syntax,omitempty"`
}

type ResponseCustomToolFormatType string

const (
	ResponseCustomToolFormatTypeText    ResponseCustomToolFormatType = "text"
	ResponseCustomToolFormatTypeGrammar ResponseCustomToolFormatType = "grammar"
)

type ResponseCustomToolFormatSyntax string

const (
	ResponseCustomToolFormatSyntaxLark  ResponseCustomToolFormatSyntax = "lark"
	ResponseCustomToolFormatSyntaxRegex ResponseCustomToolFormatSyntax = "regex"
)

type ResponseToolChoice struct {
	Option   ResponseToolChoiceOption
	Types    *ResponseToolChoiceTypesParam
	Allowed  *ResponseToolChoiceAllowedParam
	Function *ResponseToolChoiceFunctionParam
	MCP      *ResponseToolChoiceMCPParam
	Custom   *ResponseToolChoiceCustomParam
}

func (toolChoice ResponseToolChoice) MarshalJSON() ([]byte, error) {
	if toolChoice.Option != "" {
		return json.Marshal(toolChoice.Option)
	}
	if toolChoice.Types != nil {
		return json.Marshal(toolChoice.Types)
	}
	if toolChoice.Allowed != nil {
		return json.Marshal(toolChoice.Allowed)
	}
	if toolChoice.Function != nil {
		return json.Marshal(toolChoice.Function)
	}
	if toolChoice.MCP != nil {
		return json.Marshal(toolChoice.MCP)
	}
	if toolChoice.Custom != nil {
		return json.Marshal(toolChoice.Custom)
	}
	return json.Marshal(nil)
}

func (toolChoice *ResponseToolChoice) UnmarshalJSON(data []byte) error {
	for _, b := range data {
		switch b {
		case ' ', '\r', '\n', '\t':
		case 'n':
			return nil
		case '"':
			return json.Unmarshal(data, &toolChoice.Option)
		case '{':
			var ir struct {
				Type ResponseToolChoiceType `json:"type"`
			}
			if err := json.Unmarshal(data, &ir); err != nil {
				return err
			}
			switch ir.Type {
			case ResponseToolChoiceTypeAllowedTools:
				return json.Unmarshal(data, &toolChoice.Allowed)
			case ResponseToolChoiceTypeFileSearch, ResponseToolChoiceTypeWebSearch, ResponseToolChoiceTypeComputerUse, ResponseToolChoiceTypeWebSearch20250311, ResponseToolChoiceTypeImageGeneration, ResponseToolChoiceTypeCodeInterpreter:
				return json.Unmarshal(data, &toolChoice.Types)
			case ResponseToolChoiceTypeFunction:
				return json.Unmarshal(data, &toolChoice.Function)
			case ResponseToolChoiceTypeMCP:
				return json.Unmarshal(data, &toolChoice.MCP)
			case ResponseToolChoiceTypeCustom:
				return json.Unmarshal(data, &toolChoice.Custom)
			}
			return fmt.Errorf("unknown tool_choice type %q", ir.Type)
		default:
			return errors.New("tool_choice should be a string, an object or null")
		}
	}
	return errors.New("empty tool_choice")
}

type ResponseToolChoiceOption string

const (
	ChatCompletionToolChoiceOptionAuto     ResponseToolChoiceOption = "auto"
	ChatCompletionToolChoiceOptionNone     ResponseToolChoiceOption = "none"
	ChatCompletionToolChoiceOptionRequired ResponseToolChoiceOption = "required"
)

type ResponseToolChoiceType string

const (
	ResponseToolChoiceTypeFunction          ResponseToolChoiceType = "function"
	ResponseToolChoiceTypeAllowedTools      ResponseToolChoiceType = "allowed_tools"
	ResponseToolChoiceTypeFileSearch        ResponseToolChoiceType = "file_search"
	ResponseToolChoiceTypeWebSearch         ResponseToolChoiceType = "web_search_preview"
	ResponseToolChoiceTypeComputerUse       ResponseToolChoiceType = "computer_use_preview"
	ResponseToolChoiceTypeWebSearch20250311 ResponseToolChoiceType = "web_search_preview_2025_03_11"
	ResponseToolChoiceTypeImageGeneration   ResponseToolChoiceType = "image_generation"
	ResponseToolChoiceTypeCodeInterpreter   ResponseToolChoiceType = "code_interpreter"
	ResponseToolChoiceTypeMCP               ResponseToolChoiceType = "mcp"
	ResponseToolChoiceTypeCustom            ResponseToolChoiceType = "custom"
)

type ResponseToolChoiceAllowedParam struct {
	Mode  ResponseToolChoiceOption         `json:"mode"`
	Tools []*ResponseToolChoiceAllowedTool `json:"tools"`
	Type  ResponseToolChoiceType           `json:"type"`
}

type ResponseToolChoiceAllowedTool struct {
	Name string `json:"name"`
}

type ResponseToolChoiceTypesParam struct {
	Type ResponseToolChoiceType `json:"type"`
}

type ResponseToolChoiceFunctionParam struct {
	Name string                 `json:"name"`
	Type ResponseToolChoiceType `json:"type"`
}

type ResponseToolChoiceMCPParam struct {
	ServerLabel string                 `json:"server_label"`
	Type        ResponseToolChoiceType `json:"type"`
	Name        string                 `json:"name,omitempty"`
}

type ResponseToolChoiceCustomParam struct {
	Name string                 `json:"name"`
	Type ResponseToolChoiceType `json:"type"`
}

type ResponseTruncationStrategy string

const (
	ResponseTruncationStrategyAuto     ResponseTruncationStrategy = "auto"
	ResponseTruncationStrategyDisabled ResponseTruncationStrategy = "disabled"
)

type ResponseStream = iter.Seq2[Event, error]

type Response struct {
	ID                 string                     `json:"id"`
	CreatedAt          float64                    `json:"created_at"`
	Error              *ResponseError             `json:"error"`
	IncompleteDetails  *ResponseIncompleteDetails `json:"incomplete_details"`
	Instructions       ResponseInputParam         `json:"instructions"`
	Metadata           ResponseMetadata           `json:"metadata"`
	Model              string                     `json:"model"`
	Object             string                     `json:"object"`
	Output             []*ResponseOutputItem      `json:"output"`
	ParallelToolCalls  bool                       `json:"parallel_tool_calls"`
	Temperature        *float64                   `json:"temperature"`
	ToolChoice         *ResponseToolChoice        `json:"tool_choice"`
	Tools              []*ResponseToolParam       `json:"tools"`
	TopP               *float64                   `json:"top_p"`
	Background         *bool                      `json:"background"`
	Conversation       *ResponseConversation      `json:"conversation"`
	MaxOutputTokens    *int                       `json:"max_output_tokens"`
	MaxToolCalls       *int                       `json:"max_tool_calls"`
	PreviousResponseID string                     `json:"previous_response_id"`
	Prompt             *ResponsePromptParam       `json:"prompt"`
	PromptCacheKey     string                     `json:"prompt_cache_key"`
	Reasoning          *ResponseReasoning         `json:"reasoning"`
	SafetyIdentifier   string                     `json:"safety_identifier"`
	ServiceTier        ResponseServiceTier        `json:"service_tier"`
	Status             ResponseStatus             `json:"status"`
	Text               *ResponseTextConfigParam   `json:"text"`
	TopLogprobs        *int                       `json:"top_logprobs"`
	Truncation         ResponseTruncationStrategy `json:"truncation"`
	Usage              *ResponseUsage             `json:"usage"`
	User               string                     `json:"user"`
}

type ResponseOutputItem = ResponseInputItemParam

type ResponseErrorCode string

const (
	ResponseErrorCodeServerError                 ResponseErrorCode = "server_error"
	ResponseErrorCodeRateLimitExceeded           ResponseErrorCode = "rate_limit_exceeded"
	ResponseErrorCodeInvalidPrompt               ResponseErrorCode = "invalid_prompt"
	ResponseErrorCodeVectorStoreTimeout          ResponseErrorCode = "vector_store_timeout"
	ResponseErrorCodeInvalidImage                ResponseErrorCode = "invalid_image"
	ResponseErrorCodeInvalidImageFormat          ResponseErrorCode = "invalid_image_format"
	ResponseErrorCodeInvalidBase64Image          ResponseErrorCode = "invalid_base64_image"
	ResponseErrorCodeInvalidImageURL             ResponseErrorCode = "invalid_image_url"
	ResponseErrorCodeImageTooLarge               ResponseErrorCode = "image_too_large"
	ResponseErrorCodeImageTooSmall               ResponseErrorCode = "image_too_small"
	ResponseErrorCodeImageParseError             ResponseErrorCode = "image_parse_error"
	ResponseErrorCodeImageContentPolicyViolation ResponseErrorCode = "image_content_policy_violation"
	ResponseErrorCodeInvalidImageMode            ResponseErrorCode = "invalid_image_mode"
	ResponseErrorCodeImageFileTooLarge           ResponseErrorCode = "image_file_too_large"
	ResponseErrorCodeUnsupportedImageMediaType   ResponseErrorCode = "unsupported_image_media_type"
	ResponseErrorCodeEmptyImageFile              ResponseErrorCode = "empty_image_file"
	ResponseErrorCodeFailedToDownloadImage       ResponseErrorCode = "failed_to_download_image"
	ResponseErrorCodeImageFileNotFound           ResponseErrorCode = "image_file_not_found"
)

type ResponseError struct {
	Code    ResponseErrorCode `json:"code"`
	Message string            `json:"message"`
}

type ResponseIncompleteReason string

const (
	ResponseIncompleteReasonMaxOutputTokens ResponseIncompleteReason = "max_output_tokens"
	ResponseIncompleteReasonContentFilter   ResponseIncompleteReason = "content_filter"
)

type ResponseIncompleteDetails struct {
	Reason ResponseIncompleteReason `json:"reason"`
}

type ResponseInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponseOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponseUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	InputTokensDetails  *ResponseInputTokensDetails  `json:"input_tokens_details"`
	OutputTokens        int                          `json:"output_tokens"`
	OutputTokensDetails *ResponseOutputTokensDetails `json:"output_tokens_details"`
	TotalTokens         int                          `json:"total_tokens"`
}

type Event interface {
	EventType() EventType
}

type EventType string

const (
	EventTypeResponseAudioDelta                      EventType = "response.audio.delta"
	EventTypeResponseAudioDone                       EventType = "response.audio.done"
	EventTypeResponseAudioTranscriptDelta            EventType = "response.audio.transcript.delta"
	EventTypeResponseAudioTranscriptDone             EventType = "response.audio.transcript.done"
	EventTypeResponseCodeInterpreterCallCodeDelta    EventType = "response.code_interpreter_call_code.delta"
	EventTypeResponseCodeInterpreterCallCodeDone     EventType = "response.code_interpreter_call_code.done"
	EventTypeResponseCodeInterpreterCallCompleted    EventType = "response.code_interpreter_call.completed"
	EventTypeResponseCodeInterpreterCallInProgress   EventType = "response.code_interpreter_call.in_progress"
	EventTypeResponseCodeInterpreterCallInterpreting EventType = "response.code_interpreter_call.interpreting"
	EventTypeResponseCompleted                       EventType = "response.completed"
	EventTypeResponseContentPartAdded                EventType = "response.content_part.added"
	EventTypeResponseContentPartDone                 EventType = "response.content_part.done"
	EventTypeResponseCreated                         EventType = "response.created"
	EventTypeError                                   EventType = "error"
	EventTypeResponseFileSearchCallCompleted         EventType = "response.file_search_call.completed"
	EventTypeResponseFileSearchCallInProgress        EventType = "response.file_search_call.in_progress"
	EventTypeResponseFileSearchCallSearching         EventType = "response.file_search_call.searching"
	EventTypeResponseFunctionCallArgumentsDelta      EventType = "response.function_call_arguments.delta"
	EventTypeResponseFunctionCallArgumentsDone       EventType = "response.function_call_arguments.done"
	EventTypeResponseInProgress                      EventType = "response.in_progress"
	EventTypeResponseFailed                          EventType = "response.failed"
	EventTypeResponseIncomplete                      EventType = "response.incomplete"
	EventTypeResponseOutputItemAdded                 EventType = "response.output_item.added"
	EventTypeResponseOutputItemDone                  EventType = "response.output_item.done"
	EventTypeResponseReasoningSummaryPartAdded       EventType = "response.reasoning_summary_part.added"
	EventTypeResponseReasoningSummaryPartDone        EventType = "response.reasoning_summary_part.done"
	EventTypeResponseReasoningSummaryTextDelta       EventType = "response.reasoning_summary_text.delta"
	EventTypeResponseReasoningSummaryTextDone        EventType = "response.reasoning_summary_text.done"
	EventTypeResponseReasoningTextDelta              EventType = "response.reasoning_text.delta"
	EventTypeResponseReasoningTextDone               EventType = "response.reasoning_text.done"
	EventTypeResponseRefusalDelta                    EventType = "response.refusal.delta"
	EventTypeResponseRefusalDone                     EventType = "response.refusal.done"
	EventTypeResponseOutputTextDelta                 EventType = "response.output_text.delta"
	EventTypeResponseOutputTextDone                  EventType = "response.output_text.done"
	EventTypeResponseWebSearchCallCompleted          EventType = "response.web_search_call.completed"
	EventTypeResponseWebSearchCallInProgress         EventType = "response.web_search_call.in_progress"
	EventTypeResponseWebSearchCallSearching          EventType = "response.web_search_call.searching"
	EventTypeResponseImageGenCallCompleted           EventType = "response.image_generation_call.completed"
	EventTypeResponseImageGenCallGenerating          EventType = "response.image_generation_call.generating"
	EventTypeResponseImageGenCallInProgress          EventType = "response.image_generation_call.in_progress"
	EventTypeResponseImageGenCallPartialImage        EventType = "response.image_generation_call.partial_image"
	EventTypeResponseMcpCallArgumentsDelta           EventType = "response.mcp_call_arguments.delta"
	EventTypeResponseMcpCallArgumentsDone            EventType = "response.mcp_call_arguments.done"
	EventTypeResponseMcpCallCompleted                EventType = "response.mcp_call.completed"
	EventTypeResponseMcpCallFailed                   EventType = "response.mcp_call.failed"
	EventTypeResponseMcpCallInProgress               EventType = "response.mcp_call.in_progress"
	EventTypeResponseMcpListToolsCompleted           EventType = "response.mcp_list_tools.completed"
	EventTypeResponseMcpListToolsFailed              EventType = "response.mcp_list_tools.failed"
	EventTypeResponseMcpListToolsInProgress          EventType = "response.mcp_list_tools.in_progress"
	EventTypeResponseOutputTextAnnotationAdded       EventType = "response.output_text.annotation.added"
	EventTypeResponseQueued                          EventType = "response.queued"
	EventTypeResponseCustomToolCallInputDelta        EventType = "response.custom_tool_call_input.delta"
	EventTypeResponseCustomToolCallInputDone         EventType = "response.custom_tool_call_input.done"
)

type ResponseCreatedEvent struct {
	Response       Response  `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCreatedEvent) EventType() EventType { return EventTypeResponseCreated }

type ResponseCompletedEvent struct {
	Response       *Response `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCompletedEvent) EventType() EventType { return EventTypeResponseCompleted }

type ResponseInProgressEvent struct {
	Response       *Response `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseInProgressEvent) EventType() EventType { return EventTypeResponseInProgress }

type ResponseFailedEvent struct {
	Response       *Response `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFailedEvent) EventType() EventType { return EventTypeResponseFailed }

type ResponseIncompleteEvent struct {
	Response       *Response `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseIncompleteEvent) EventType() EventType { return EventTypeResponseIncomplete }

type ResponseQueuedEvent struct {
	Response       *Response `json:"response"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseQueuedEvent) EventType() EventType { return EventTypeResponseQueued }

type ResponseErrorEvent struct {
	Code           string    `json:"code,omitempty"`
	Message        string    `json:"message"`
	Param          string    `json:"param,omitempty"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseErrorEvent) EventType() EventType { return EventTypeError }

type ResponseTextDeltaEvent struct {
	ContentIndex   int                `json:"content_index"`
	Delta          string             `json:"delta"`
	ItemID         string             `json:"item_id"`
	Logprobs       []*ResponseLogprob `json:"logprobs"`
	OutputIndex    int                `json:"output_index"`
	SequenceNumber int                `json:"sequence_number"`
	Type           EventType          `json:"type"`
}

func (e ResponseTextDeltaEvent) EventType() EventType { return EventTypeResponseOutputTextDelta }

type ResponseTextDoneEvent struct {
	ContentIndex   int                `json:"content_index"`
	ItemID         string             `json:"item_id"`
	Logprobs       []*ResponseLogprob `json:"logprobs"`
	OutputIndex    int                `json:"output_index"`
	SequenceNumber int                `json:"sequence_number"`
	Text           string             `json:"text"`
	Type           EventType          `json:"type"`
}

func (e ResponseTextDoneEvent) EventType() EventType { return EventTypeResponseOutputTextDone }

type ResponseRefusalDeltaEvent struct {
	ContentIndex   int       `json:"content_index"`
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseRefusalDeltaEvent) EventType() EventType { return EventTypeResponseRefusalDelta }

type ResponseRefusalDoneEvent struct {
	ContentIndex   int       `json:"content_index"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	Refusal        string    `json:"refusal"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseRefusalDoneEvent) EventType() EventType { return EventTypeResponseRefusalDone }

type ResponseReasoningTextDeltaEvent struct {
	ContentIndex   int       `json:"content_index"`
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseReasoningTextDeltaEvent) EventType() EventType {
	return EventTypeResponseReasoningTextDelta
}

type ResponseReasoningTextDoneEvent struct {
	ContentIndex   int       `json:"content_index"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Text           string    `json:"text"`
	Type           EventType `json:"type"`
}

func (e ResponseReasoningTextDoneEvent) EventType() EventType {
	return EventTypeResponseReasoningTextDone
}

type ResponseReasoningSummaryPartAddedEvent struct {
	ItemID         string                    `json:"item_id"`
	OutputIndex    int                       `json:"output_index"`
	Part           *ResponseReasoningContent `json:"part"`
	SequenceNumber int                       `json:"sequence_number"`
	SummaryIndex   int                       `json:"summary_index"`
	Type           EventType                 `json:"type"`
}

func (e ResponseReasoningSummaryPartAddedEvent) EventType() EventType {
	return EventTypeResponseReasoningSummaryPartAdded
}

type ResponseReasoningSummaryPartDoneEvent struct {
	ItemID         string                    `json:"item_id"`
	OutputIndex    int                       `json:"output_index"`
	Part           *ResponseReasoningContent `json:"part"`
	SequenceNumber int                       `json:"sequence_number"`
	SummaryIndex   int                       `json:"summary_index"`
	Type           EventType                 `json:"type"`
}

func (e ResponseReasoningSummaryPartDoneEvent) EventType() EventType {
	return EventTypeResponseReasoningSummaryPartDone
}

type ResponseReasoningSummaryTextDeltaEvent struct {
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	SummaryIndex   int       `json:"summary_index"`
	Type           EventType `json:"type"`
}

func (e ResponseReasoningSummaryTextDeltaEvent) EventType() EventType {
	return EventTypeResponseReasoningSummaryTextDelta
}

type ResponseReasoningSummaryTextDoneEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	SummaryIndex   int       `json:"summary_index"`
	Text           string    `json:"text"`
	Type           EventType `json:"type"`
}

func (e ResponseReasoningSummaryTextDoneEvent) EventType() EventType {
	return EventTypeResponseReasoningSummaryTextDone
}

type ResponseContentPartAddedEvent struct {
	ContentIndex   int                         `json:"content_index"`
	ItemID         string                      `json:"item_id"`
	OutputIndex    int                         `json:"output_index"`
	Part           *ResponseMessageContentText `json:"part"`
	SequenceNumber int                         `json:"sequence_number"`
	Type           EventType                   `json:"type"`
}

func (e ResponseContentPartAddedEvent) EventType() EventType {
	return EventTypeResponseContentPartAdded
}

type ResponseContentPartDoneEvent struct {
	ContentIndex   int                         `json:"content_index"`
	ItemID         string                      `json:"item_id"`
	OutputIndex    int                         `json:"output_index"`
	Part           *ResponseMessageContentText `json:"part"`
	SequenceNumber int                         `json:"sequence_number"`
	Type           EventType                   `json:"type"`
}

func (e ResponseContentPartDoneEvent) EventType() EventType { return EventTypeResponseContentPartDone }

type ResponseOutputItemAddedEvent struct {
	Item           *ResponseInputItemParam `json:"item"`
	OutputIndex    int                     `json:"output_index"`
	SequenceNumber int                     `json:"sequence_number"`
	Type           EventType               `json:"type"`
}

func (e ResponseOutputItemAddedEvent) EventType() EventType { return EventTypeResponseOutputItemAdded }

type ResponseOutputItemDoneEvent struct {
	Item           *ResponseInputItemParam `json:"item"`
	OutputIndex    int                     `json:"output_index"`
	SequenceNumber int                     `json:"sequence_number"`
	Type           EventType               `json:"type"`
}

func (e ResponseOutputItemDoneEvent) EventType() EventType { return EventTypeResponseOutputItemDone }

type ResponseOutputTextAnnotationAddedEvent struct {
	Annotation      *ResponseAnnotation `json:"annotation"`
	AnnotationIndex int                 `json:"annotation_index"`
	ContentIndex    int                 `json:"content_index"`
	ItemID          string              `json:"item_id"`
	OutputIndex     int                 `json:"output_index"`
	SequenceNumber  int                 `json:"sequence_number"`
	Type            EventType           `json:"type"`
}

func (e ResponseOutputTextAnnotationAddedEvent) EventType() EventType {
	return EventTypeResponseOutputTextAnnotationAdded
}

type ResponseFunctionCallArgumentsDeltaEvent struct {
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFunctionCallArgumentsDeltaEvent) EventType() EventType {
	return EventTypeResponseFunctionCallArgumentsDelta
}

type ResponseFunctionCallArgumentsDoneEvent struct {
	Arguments      string    `json:"arguments"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFunctionCallArgumentsDoneEvent) EventType() EventType {
	return EventTypeResponseFunctionCallArgumentsDone
}

type ResponseCodeInterpreterCallCodeDeltaEvent struct {
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCodeInterpreterCallCodeDeltaEvent) EventType() EventType {
	return EventTypeResponseCodeInterpreterCallCodeDelta
}

type ResponseCodeInterpreterCallCodeDoneEvent struct {
	Code           string    `json:"code"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCodeInterpreterCallCodeDoneEvent) EventType() EventType {
	return EventTypeResponseCodeInterpreterCallCodeDone
}

type ResponseCodeInterpreterCallInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCodeInterpreterCallInProgressEvent) EventType() EventType {
	return EventTypeResponseCodeInterpreterCallInProgress
}

type ResponseCodeInterpreterCallCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCodeInterpreterCallCompletedEvent) EventType() EventType {
	return EventTypeResponseCodeInterpreterCallCompleted
}

type ResponseCodeInterpreterCallInterpretingEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCodeInterpreterCallInterpretingEvent) EventType() EventType {
	return EventTypeResponseCodeInterpreterCallInterpreting
}

type ResponseFileSearchCallInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFileSearchCallInProgressEvent) EventType() EventType {
	return EventTypeResponseFileSearchCallInProgress
}

type ResponseFileSearchCallCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFileSearchCallCompletedEvent) EventType() EventType {
	return EventTypeResponseFileSearchCallCompleted
}

type ResponseFileSearchCallSearchingEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseFileSearchCallSearchingEvent) EventType() EventType {
	return EventTypeResponseFileSearchCallSearching
}

type ResponseWebSearchCallInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseWebSearchCallInProgressEvent) EventType() EventType {
	return EventTypeResponseWebSearchCallInProgress
}

type ResponseWebSearchCallCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseWebSearchCallCompletedEvent) EventType() EventType {
	return EventTypeResponseWebSearchCallCompleted
}

type ResponseWebSearchCallSearchingEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseWebSearchCallSearchingEvent) EventType() EventType {
	return EventTypeResponseWebSearchCallSearching
}

type ResponseImageGenCallInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseImageGenCallInProgressEvent) EventType() EventType {
	return EventTypeResponseImageGenCallInProgress
}

type ResponseImageGenCallGeneratingEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseImageGenCallGeneratingEvent) EventType() EventType {
	return EventTypeResponseImageGenCallGenerating
}

type ResponseImageGenCallCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseImageGenCallCompletedEvent) EventType() EventType {
	return EventTypeResponseImageGenCallCompleted
}

type ResponseImageGenCallPartialImageEvent struct {
	ItemID            string    `json:"item_id"`
	OutputIndex       int       `json:"output_index"`
	PartialImageB64   string    `json:"partial_image_b64"`
	PartialImageIndex int       `json:"partial_image_index"`
	SequenceNumber    int       `json:"sequence_number"`
	Type              EventType `json:"type"`
}

func (e ResponseImageGenCallPartialImageEvent) EventType() EventType {
	return EventTypeResponseImageGenCallPartialImage
}

type ResponseMcpCallArgumentsDeltaEvent struct {
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpCallArgumentsDeltaEvent) EventType() EventType {
	return EventTypeResponseMcpCallArgumentsDelta
}

type ResponseMcpCallArgumentsDoneEvent struct {
	Arguments      string    `json:"arguments"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpCallArgumentsDoneEvent) EventType() EventType {
	return EventTypeResponseMcpCallArgumentsDone
}

type ResponseMcpCallInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpCallInProgressEvent) EventType() EventType {
	return EventTypeResponseMcpCallInProgress
}

type ResponseMcpCallCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpCallCompletedEvent) EventType() EventType {
	return EventTypeResponseMcpCallCompleted
}

type ResponseMcpCallFailedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpCallFailedEvent) EventType() EventType {
	return EventTypeResponseMcpCallFailed
}

type ResponseMcpListToolsInProgressEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpListToolsInProgressEvent) EventType() EventType {
	return EventTypeResponseMcpListToolsInProgress
}

type ResponseMcpListToolsCompletedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpListToolsCompletedEvent) EventType() EventType {
	return EventTypeResponseMcpListToolsCompleted
}

type ResponseMcpListToolsFailedEvent struct {
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseMcpListToolsFailedEvent) EventType() EventType {
	return EventTypeResponseMcpListToolsFailed
}

type ResponseCustomToolCallInputDeltaEvent struct {
	Delta          string    `json:"delta"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCustomToolCallInputDeltaEvent) EventType() EventType {
	return EventTypeResponseCustomToolCallInputDelta
}

type ResponseCustomToolCallInputDoneEvent struct {
	Input          string    `json:"input"`
	ItemID         string    `json:"item_id"`
	OutputIndex    int       `json:"output_index"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseCustomToolCallInputDoneEvent) EventType() EventType {
	return EventTypeResponseCustomToolCallInputDone
}

type ResponseAudioDeltaEvent struct {
	Delta          string    `json:"delta"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseAudioDeltaEvent) EventType() EventType { return EventTypeResponseAudioDelta }

type ResponseAudioDoneEvent struct {
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseAudioDoneEvent) EventType() EventType { return EventTypeResponseAudioDone }

type ResponseAudioTranscriptDeltaEvent struct {
	Delta          string    `json:"delta"`
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseAudioTranscriptDeltaEvent) EventType() EventType {
	return EventTypeResponseAudioTranscriptDelta
}

type ResponseAudioTranscriptDoneEvent struct {
	SequenceNumber int       `json:"sequence_number"`
	Type           EventType `json:"type"`
}

func (e ResponseAudioTranscriptDoneEvent) EventType() EventType {
	return EventTypeResponseAudioTranscriptDone
}

func NewResponseBuilder() *ResponseBuilder {
	return &ResponseBuilder{
		Response: &Response{},
	}
}

type ResponseBuilder struct {
	Response *Response
}

func (builder *ResponseBuilder) Build() *Response {
	return builder.Response
}

func (builder *ResponseBuilder) Add(event Event) {
	if event.EventType() == EventTypeError {
		panic("error event")
	}
	if event.EventType() == EventTypeResponseCompleted {
		if completedResponse, ok := event.(*ResponseCompletedEvent); ok {
			builder.Response = completedResponse.Response
		}
	}
}
