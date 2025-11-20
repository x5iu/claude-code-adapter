package adapter

import (
	"fmt"
	"strings"

	"github.com/samber/lo"
	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
)

func init() {
	viper.SetDefault(delimiter.ViperKey("options", "strict"), false)
	viper.SetDefault(delimiter.ViperKey("options", "reasoning", "format"), string(openrouter.ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1))
	viper.SetDefault(delimiter.ViperKey("options", "reasoning", "effort"), "")
	viper.SetDefault(delimiter.ViperKey("options", "models"), map[string]string{})
}

type ConvertRequestOptions struct {
}

type ConvertRequestOption func(*ConvertRequestOptions)

func ConvertAnthropicRequestToOpenRouterRequest(
	src *anthropic.GenerateMessageRequest,
	options ...ConvertRequestOption,
) (dst *openrouter.CreateChatCompletionRequest) {
	convertOptions := &ConvertRequestOptions{}
	for _, applyOption := range options {
		applyOption(convertOptions)
	}
	dst = &openrouter.CreateChatCompletionRequest{
		Model:       src.Model,
		MaxTokens:   lo.ToPtr(src.MaxTokens),
		Temperature: lo.ToPtr(src.Temperature),
		TopK:        src.TopK,
		TopP:        src.TopP,
		Usage:       &openrouter.ChatCompletionUsageOptions{Include: true},
	}
	if modelMapper := viper.GetStringMapString(delimiter.ViperKey("options", "models")); modelMapper != nil {
		if targetModel, ok := modelMapper[dst.Model]; ok {
			dst.Model = targetModel
		}
	}
	if metadata := src.Metadata; metadata != nil && metadata.UserID != "" {
		dst.User = metadata.UserID
	}
	if len(src.StopSequences) > 0 {
		dst.Stop = src.StopSequences
	}
	if srcToolChoice := src.ToolChoice; srcToolChoice != nil {
		dst.ParallelToolCalls = lo.ToPtr(!srcToolChoice.DisableParallelToolUse)
		var dstToolChoice *openrouter.ChatCompletionToolChoice
		switch srcToolChoice.Type {
		case anthropic.ToolChoiceTypeTool:
			dstToolChoice = &openrouter.ChatCompletionToolChoice{
				Tool: &openrouter.ChatCompletionTool{
					Type:     openrouter.ChatCompletionMessageToolCallTypeFunction,
					Function: &openrouter.ChatCompletionFunction{Name: srcToolChoice.Name},
				},
			}
		case anthropic.ToolChoiceTypeAuto:
			dstToolChoice = &openrouter.ChatCompletionToolChoice{
				Mode: openrouter.ChatCompletionToolChoiceTypeAuto,
			}
		case anthropic.ToolChoiceTypeNone:
			dstToolChoice = &openrouter.ChatCompletionToolChoice{
				Mode: openrouter.ChatCompletionToolChoiceTypeNone,
			}
		case anthropic.ToolChoiceTypeAny:
			dstToolChoice = &openrouter.ChatCompletionToolChoice{
				Mode: openrouter.ChatCompletionToolChoiceTypeRequired,
			}
		}
		dst.ToolChoice = dstToolChoice
	}
	if len(src.Tools) > 0 {
		dst.Tools = make([]*openrouter.ChatCompletionTool, 0, len(src.Tools))
		for _, srcTool := range src.Tools {
			var srcToolType anthropic.ToolType
			// A custom tool can omit the type parameter, so we consider a tool with a null type value to be a custom tool.
			// reference: https://docs.anthropic.com/en/api/messages#custom-tool
			if srcTool.Type == nil {
				srcToolType = anthropic.ToolTypeCustom
			} else {
				srcToolType = *srcTool.Type
			}
			switch srcToolType {
			case anthropic.ToolTypeCustom:
				dstTool := &openrouter.ChatCompletionTool{
					Type: openrouter.ChatCompletionMessageToolCallTypeFunction,
					Function: &openrouter.ChatCompletionFunction{
						Name:        srcTool.Name,
						Description: srcTool.Description,
						Strict:      viper.GetBool(delimiter.ViperKey("options", "strict")),
						Parameters:  openrouter.ChatCompletionJSONSchemaObject(srcTool.InputSchema),
					},
				}
				if srcCacheControl := srcTool.CacheControl; srcCacheControl != nil {
					// Anthropic's Tool supports the CacheControl parameter, but in OpenRouter's Tool definition we have not yet
					// found a field for setting CacheControl. Therefore, we will temporarily ignore the CacheControl setting and add
					// it in the future.
				}
				dst.Tools = append(dst.Tools, dstTool)
			}
		}
	}
	if thinking := src.Thinking; thinking != nil {
		reasoning := &openrouter.ChatCompletionReasoning{
			MaxTokens: thinking.BudgetTokens,
		}
		switch thinking.Type {
		case anthropic.ThinkingTypeEnabled:
			reasoning.Enabled = true
		case anthropic.ThinkingTypeDisabled:
			reasoning.Enabled = false
		}
		dst.Reasoning = reasoning
	}
	switch format := getOpenRouterModelReasoningFormat(dst.Model); format {
	case openrouter.ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1:
		if dst.Reasoning == nil {
			if viper.GetBool(delimiter.ViperKey("anthropic", "force_thinking")) {
				if dst.MaxTokens == nil || *dst.MaxTokens <= 1024 {
					dst.MaxTokens = lo.ToPtr(32 * 1024)
				}
				dst.Reasoning = &openrouter.ChatCompletionReasoning{
					Enabled:   true,
					MaxTokens: *dst.MaxTokens - 1,
				}
			}
		}
	case openrouter.ChatCompletionMessageReasoningDetailFormatOpenAIResponsesV1:
		var effort openrouter.ChatCompletionReasoningEffort
		if model, suffix, ok := strings.Cut(dst.Model, ":"); ok {
			dst.Model = model
			effort = openrouter.ChatCompletionReasoningEffort(suffix)
		} else {
			effort = openrouter.ChatCompletionReasoningEffort(viper.GetString(delimiter.ViperKey("options", "reasoning", "effort")))
		}
		if !effort.IsEmpty() {
			if dst.Reasoning == nil {
				dst.Reasoning = &openrouter.ChatCompletionReasoning{
					Effort: effort,
				}
			} else {
				dst.Reasoning.MaxTokens = 0
				dst.Reasoning.Effort = effort
			}
		}
	case openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1:
		// Google: Reasoning is mandatory for this endpoint and cannot be disabled.
		if dst.Reasoning == nil {
			dst.Reasoning = &openrouter.ChatCompletionReasoning{Enabled: true}
		} else {
			dst.Reasoning.Enabled = true
		}
	}
	dstMessages := make([]*openrouterChatCompletionMessageWrapper, 0, len(src.Messages))
	if len(src.System) > 0 {
		dstSystemMessage := &openrouter.ChatCompletionMessage{
			Role: openrouter.ChatCompletionMessageRoleSystem,
			Content: &openrouter.ChatCompletionMessageContent{
				Type:  openrouter.ChatCompletionMessageContentTypeParts,
				Parts: make([]*openrouter.ChatCompletionMessageContentPart, 0, len(src.System)),
			},
		}
		for _, systemContent := range src.System {
			switch systemContent.Type {
			case anthropic.MessageContentTypeText:
				dstPart := &openrouter.ChatCompletionMessageContentPart{
					Type: openrouter.ChatCompletionMessageContentPartTypeText,
					Text: systemContent.Text,
				}
				if srcCacheControl := systemContent.CacheControl; srcCacheControl != nil {
					dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
						Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
						TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
					}
				}
				dstSystemMessage.Content.Parts = append(dstSystemMessage.Content.Parts, dstPart)
			case anthropic.MessageContentTypeImage:
				if srcSystemContentSource := systemContent.Source; srcSystemContentSource != nil {
					dstPart := &openrouter.ChatCompletionMessageContentPart{
						Type: openrouter.ChatCompletionMessageContentPartTypeImage,
						ImageUrl: &openrouter.ChatCompletionMessageContentPartImageUrl{
							Url: fmt.Sprintf("data:%s;%s,%s", srcSystemContentSource.MediaType, srcSystemContentSource.Type, srcSystemContentSource.Data),
						},
					}
					if srcCacheControl := systemContent.CacheControl; srcCacheControl != nil {
						dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
							Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
							TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
						}
					}
					dstSystemMessage.Content.Parts = append(dstSystemMessage.Content.Parts, dstPart)
				}
			}
		}
		if len(dstSystemMessage.Content.Parts) > 0 {
			dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
				ChatCompletionMessage: dstSystemMessage,
			})
		}
	}
	for _, srcMessage := range src.Messages {
		var dstRole openrouter.ChatCompletionRole
		switch srcMessage.Role {
		case anthropic.MessageRoleUser:
			dstRole = openrouter.ChatCompletionMessageRoleUser
		case anthropic.MessageRoleAssistant:
			dstRole = openrouter.ChatCompletionMessageRoleAssistant
		}
		for _, srcMessageContent := range srcMessage.Content {
			switch srcMessageContent.Type {
			case anthropic.MessageContentTypeThinking:
				dstMessage := &openrouter.ChatCompletionMessage{
					Role:      dstRole,
					Reasoning: srcMessageContent.Thinking,
					ReasoningDetails: []*openrouter.ChatCompletionMessageReasoningDetail{
						{
							Type:      openrouter.ChatCompletionMessageReasoningDetailTypeReasoningText,
							Text:      srcMessageContent.Thinking,
							Signature: srcMessageContent.Signature,
							Format:    openrouter.ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1,
						},
					},
				}
				dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
					ChatCompletionMessage:      dstMessage,
					underlyingAnthropicMessage: srcMessage,
				})
			case anthropic.MessageContentTypeRedactedThinking:
				panic("unreachable redacted_thinking")
			case anthropic.MessageContentTypeToolUse:
				dstMessage := &openrouter.ChatCompletionMessage{
					Role: openrouter.ChatCompletionMessageRoleAssistant,
					ToolCalls: []*openrouter.ChatCompletionToolCall{
						{
							ID:   srcMessageContent.ID,
							Type: openrouter.ChatCompletionMessageToolCallTypeFunction,
							Function: &openrouter.ChatCompletionMessageToolCallFunction{
								Name:      srcMessageContent.Name,
								Arguments: string(srcMessageContent.Input),
							},
						},
					},
				}
				if srcCacheControl := srcMessageContent.CacheControl; srcCacheControl != nil {
					// Anthropic's ToolUse supports the CacheControl parameter, but in OpenRouter's ToolCalls definition we have not yet
					// found a field for setting CacheControl. Therefore, we will temporarily ignore the CacheControl setting and add
					// it in the future.
				}
				dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
					ChatCompletionMessage:      dstMessage,
					underlyingAnthropicMessage: srcMessage,
				})
			case anthropic.MessageContentTypeToolResult:
				dstMessage := &openrouter.ChatCompletionMessage{
					Role:       openrouter.ChatCompletionMessageRoleTool,
					ToolCallID: srcMessageContent.ToolUseID,
				}
				if srcMessageContent.Content != nil {
					dstMessage.Content = convertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent(srcMessageContent.Content)
				}
				dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
					ChatCompletionMessage:      dstMessage,
					underlyingAnthropicMessage: srcMessage,
				})
			case anthropic.MessageContentTypeText:
				dstPart := &openrouter.ChatCompletionMessageContentPart{
					Type: openrouter.ChatCompletionMessageContentPartTypeText,
					Text: srcMessageContent.Text,
				}
				if srcCacheControl := srcMessageContent.CacheControl; srcCacheControl != nil {
					dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
						Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
						TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
					}
				}
				dstMessage := &openrouter.ChatCompletionMessage{
					Role: dstRole,
					Content: &openrouter.ChatCompletionMessageContent{
						Type:  openrouter.ChatCompletionMessageContentTypeParts,
						Parts: []*openrouter.ChatCompletionMessageContentPart{dstPart},
					},
				}
				dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
					ChatCompletionMessage:      dstMessage,
					underlyingAnthropicMessage: srcMessage,
				})
			case anthropic.MessageContentTypeImage:
				if srcMessageContentSource := srcMessageContent.Source; srcMessageContentSource != nil {
					dstPart := &openrouter.ChatCompletionMessageContentPart{
						Type: openrouter.ChatCompletionMessageContentPartTypeImage,
						ImageUrl: &openrouter.ChatCompletionMessageContentPartImageUrl{
							Url: fmt.Sprintf("data:%s;%s,%s", srcMessageContentSource.MediaType, srcMessageContentSource.Type, srcMessageContentSource.Data),
						},
					}
					// Images: Content blocks in the messages.content array, in user turns
					// reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached
					if srcMessage.Role == anthropic.MessageRoleUser {
						if srcCacheControl := srcMessageContent.CacheControl; srcCacheControl != nil {
							dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
								Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
								TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
							}
						}
					}
					dstMessage := &openrouter.ChatCompletionMessage{
						Role: dstRole,
						Content: &openrouter.ChatCompletionMessageContent{
							Type:  openrouter.ChatCompletionMessageContentTypeParts,
							Parts: []*openrouter.ChatCompletionMessageContentPart{dstPart},
						},
					}
					dstMessages = append(dstMessages, &openrouterChatCompletionMessageWrapper{
						ChatCompletionMessage:      dstMessage,
						underlyingAnthropicMessage: srcMessage,
					})
				}
			}
		}
	}
	dst.Messages = canonicalOpenRouterMessages(dst.Model, dstMessages)
	return dst
}

type openrouterChatCompletionMessageWrapper struct {
	*openrouter.ChatCompletionMessage
	underlyingAnthropicMessage *anthropic.Message
}

func canonicalOpenRouterMessages(
	model string,
	messageWrappers []*openrouterChatCompletionMessageWrapper,
) (messages []*openrouter.ChatCompletionMessage) {
	messages = make([]*openrouter.ChatCompletionMessage, 0, len(messageWrappers))
	var (
		currentChatCompletionMessage      *openrouter.ChatCompletionMessage
		currentUnderlyingAnthropicMessage *anthropic.Message
	)
	for _, wrapper := range messageWrappers {
		if wrapper.Role == openrouter.ChatCompletionMessageRoleSystem ||
			wrapper.Role == openrouter.ChatCompletionMessageRoleTool {
			if currentChatCompletionMessage != nil {
				messages = append(messages, currentChatCompletionMessage)
				currentChatCompletionMessage = nil
			}
			messages = append(messages, wrapper.ChatCompletionMessage)
		} else if underlyingMessage := wrapper.underlyingAnthropicMessage; underlyingMessage != nil {
			if underlyingMessage != currentUnderlyingAnthropicMessage {
				currentUnderlyingAnthropicMessage = underlyingMessage
				if currentChatCompletionMessage != nil {
					messages = append(messages, currentChatCompletionMessage)
				}
				currentChatCompletionMessage = wrapper.ChatCompletionMessage
			} else if currentChatCompletionMessage != nil {
				if currentChatCompletionMessage.Role == openrouter.ChatCompletionMessageRoleAssistant {
					if currentChatCompletionMessage.Reasoning == "" && wrapper.Reasoning != "" {
						currentChatCompletionMessage.Reasoning = wrapper.Reasoning
					}
					if len(wrapper.ReasoningDetails) > 0 {
						currentChatCompletionMessage.ReasoningDetails = append(currentChatCompletionMessage.ReasoningDetails, wrapper.ReasoningDetails...)
					}
					if len(wrapper.ToolCalls) > 0 {
						currentChatCompletionMessage.ToolCalls = append(currentChatCompletionMessage.ToolCalls, wrapper.ToolCalls...)
					}
				}
				if currentChatCompletionMessage.Content != nil && wrapper.Content != nil {
					if currentChatCompletionMessage.Content.IsText() {
						currentChatCompletionMessage.Content = &openrouter.ChatCompletionMessageContent{
							Type: openrouter.ChatCompletionMessageContentTypeParts,
							Parts: []*openrouter.ChatCompletionMessageContentPart{
								{
									Type: openrouter.ChatCompletionMessageContentPartTypeText,
									Text: currentChatCompletionMessage.Content.Text,
								},
							},
						}
					}
					switch wrapper.Content.Type {
					case openrouter.ChatCompletionMessageContentTypeText:
						currentChatCompletionMessage.Content.Parts = append(currentChatCompletionMessage.Content.Parts, &openrouter.ChatCompletionMessageContentPart{
							Type: openrouter.ChatCompletionMessageContentPartTypeText,
							Text: wrapper.Content.Text,
						})
					case openrouter.ChatCompletionMessageContentTypeParts:
						currentChatCompletionMessage.Content.Parts = append(currentChatCompletionMessage.Content.Parts, wrapper.Content.Parts...)
					}
				}
			}
		}
	}
	if currentChatCompletionMessage != nil {
		messages = append(messages, currentChatCompletionMessage)
	}
	for _, message := range messages {
		switch message.Role {
		case openrouter.ChatCompletionMessageRoleAssistant:
			if content := message.Content; content != nil && content.IsParts() && len(content.Parts) == 1 {
				if part := content.Parts[0]; part.Type == openrouter.ChatCompletionMessageContentPartTypeText {
					// Anthropic only accepts text message in a single assistant message, and content should be a string
					content.Type = openrouter.ChatCompletionMessageContentTypeText
					content.Text = part.Text
					content.Parts = nil
				}
			}
			if len(message.ReasoningDetails) > 0 {
				for index, reasoningDetail := range message.ReasoningDetails {
					reasoningDetail.Index = index
				}
				switch format := getOpenRouterModelReasoningFormat(model); format {
				case openrouter.ChatCompletionMessageReasoningDetailFormatOpenAIResponsesV1,
					openrouter.ChatCompletionMessageReasoningDetailFormatGoogleGeminiV1:
					revisedReasoningDetails := make([]*openrouter.ChatCompletionMessageReasoningDetail, 0, len(message.ReasoningDetails))
					for _, reasoningDetail := range message.ReasoningDetails {
						reasoningDetail.Format = format
						if reasoningDetail.Text != "" {
							reasoningDetail.Type = openrouter.ChatCompletionMessageReasoningDetailTypeSummary
						}
						signature := reasoningDetail.Signature
						reasoningDetail.Signature = ""
						revisedReasoningDetails = append(revisedReasoningDetails, reasoningDetail)
						if signature != "" {
							derivedReasoningDetail := &openrouter.ChatCompletionMessageReasoningDetail{
								Type:   openrouter.ChatCompletionMessageReasoningDetailTypeEncrypted,
								Format: format,
								Index:  reasoningDetail.Index,
							}
							if delimiterIndex := strings.Index(signature, viper.GetString(delimiter.ViperKey("options", "reasoning", "delimiter"))); delimiterIndex != -1 {
								derivedReasoningDetail.ID = signature[:delimiterIndex]
								derivedReasoningDetail.Data = signature[delimiterIndex+1:]
							} else {
								derivedReasoningDetail.Data = signature
							}
							revisedReasoningDetails = append(revisedReasoningDetails, derivedReasoningDetail)
						}
					}
					message.ReasoningDetails = revisedReasoningDetails
				}
			}
			if len(message.ToolCalls) > 0 {
				for index, toolCall := range message.ToolCalls {
					toolCall.Index = index
				}
			}
		}
		// OpenRouter says that the cache_control breakpoint can only be inserted into the text part of a multipart message.
		// So we remove CacheControl fields from content parts for non-text type.
		//
		// reference: https://openrouter.ai/docs/features/prompt-caching#anthropic-claude
		if content := message.Content; content != nil && content.IsParts() {
			for _, part := range content.Parts {
				if part.Type != openrouter.ChatCompletionMessageContentPartTypeText {
					part.CacheControl = nil
				}
			}
		}
	}
	return messages
}

func convertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent(
	src anthropic.MessageContents,
) (dst *openrouter.ChatCompletionMessageContent) {
	if len(src) == 0 {
		return &openrouter.ChatCompletionMessageContent{
			Type: openrouter.ChatCompletionMessageContentTypeText,
		}
	}
	dst = &openrouter.ChatCompletionMessageContent{
		Type:  openrouter.ChatCompletionMessageContentTypeParts,
		Parts: make([]*openrouter.ChatCompletionMessageContentPart, 0, len(src)),
	}
	for _, srcContent := range src {
		switch srcContent.Type {
		case anthropic.MessageContentTypeToolUse, anthropic.MessageContentTypeToolResult:
			panic("unreachable tool_use/tool_result")
		case anthropic.MessageContentTypeThinking, anthropic.MessageContentTypeRedactedThinking:
			panic("unreachable thinking/redacted_thinking")
		case anthropic.MessageContentTypeText:
			dstPart := &openrouter.ChatCompletionMessageContentPart{
				Type: openrouter.ChatCompletionMessageContentPartTypeText,
				Text: srcContent.Text,
			}
			// No idea why Claude Code send empty text in tool_result, so we replace it with a hint message if necessary.
			if viper.GetBool(delimiter.ViperKey("options", "prevent_empty_text_tool_result")) && srcContent.Text == "" {
				dstPart.Text = "(No content)"
			}
			if srcCacheControl := srcContent.CacheControl; srcCacheControl != nil {
				dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
					Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
					TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
				}
			}
			dst.Parts = append(dst.Parts, dstPart)
		case anthropic.MessageContentTypeImage:
			if srcContentSource := srcContent.Source; srcContentSource != nil {
				dstPart := &openrouter.ChatCompletionMessageContentPart{
					Type: openrouter.ChatCompletionMessageContentPartTypeImage,
					ImageUrl: &openrouter.ChatCompletionMessageContentPartImageUrl{
						Url: fmt.Sprintf("data:%s;%s,%s", srcContentSource.MediaType, srcContentSource.Type, srcContentSource.Data),
					},
				}
				if srcCacheControl := srcContent.CacheControl; srcCacheControl != nil {
					dstPart.CacheControl = &openrouter.ChatCompletionMessageCacheControl{
						Type: openrouter.ChatCompletionMessageCacheControlType(srcCacheControl.Type),
						TTL:  openrouter.ChatCompletionMessageCacheControlTTL(srcCacheControl.TTL),
					}
				}
				dst.Parts = append(dst.Parts, dstPart)
			}
		}
	}
	return dst
}

func getOpenRouterModelReasoningFormat(
	model string,
) (format openrouter.ChatCompletionMessageReasoningDetailFormat) {
	if modelReasoningFormat := viper.GetStringMapString(delimiter.ViperKey("openrouter", "model_reasoning_format")); modelReasoningFormat != nil {
		if format, ok := modelReasoningFormat[model]; ok {
			return openrouter.ChatCompletionMessageReasoningDetailFormat(format)
		}
	}
	return openrouter.ChatCompletionMessageReasoningDetailFormat(viper.GetString(delimiter.ViperKey("options", "reasoning", "format")))
}
