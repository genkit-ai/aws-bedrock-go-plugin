// Copyright 2025 Xavier Portilla Edo
// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithydoc "github.com/aws/smithy-go/document"
	"github.com/firebase/genkit/go/ai"
)

// generateText handles text generation using Bedrock Converse API
func (b *Bedrock) generateText(ctx context.Context, modelName string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Convert Genkit request to Bedrock Converse input
	converseInput, err := b.buildConverseInput(modelName, input)
	if err != nil {
		return nil, fmt.Errorf("failed to build converse input: %w", err)
	}

	// Handle streaming vs non-streaming
	if cb != nil {
		return b.generateTextStream(ctx, converseInput, input, cb)
	}
	return b.generateTextSync(ctx, converseInput, input)
}

func (b *Bedrock) buildConverseInput(modelName string, input *ai.ModelRequest) (*bedrockruntime.ConverseInput, error) {
	if input == nil {
		return nil, fmt.Errorf("model request is nil")
	}

	converseInput := &bedrockruntime.ConverseInput{
		ModelId: aws.String(modelName),
	}

	// Convert messages
	if len(input.Messages) > 0 {
		var messages []types.Message
		var systemPrompts []types.SystemContentBlock

		for _, msg := range input.Messages {
			switch msg.Role {
			case ai.RoleSystem:
				// System messages go into separate field
				for _, part := range msg.Content {
					if part.IsText() {
						systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
							Value: part.Text,
						})
					} else if part.IsCustom() {
						// Handle custom parts, the plugin currently supports NewCachePointPart
						if cpt, ok := CachePointType(part); ok {
							systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberCachePoint{
								Value: types.CachePointBlock{
									Type: cpt,
								},
							})
						}
					}
				}
			case ai.RoleUser, ai.RoleModel, ai.RoleTool:
				// Convert message content
				var contentBlocks []types.ContentBlock
				for _, part := range msg.Content {
					if part.IsText() {
						contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{
							Value: part.Text,
						})
					} else if part.IsMedia() {
						// Handle media parts for multimodal models
						mediaType := part.ContentType

						// Parse data URL or direct content
						content := part.Text
						if strings.HasPrefix(content, "data:") {
							// Handle data URL format: data:image/png;base64,... or data:application/pdf;base64,...
							parts := strings.Split(content, ",")
							if len(parts) == 2 {
								// Extract the actual base64 data
								content = parts[1]
								// Extract MIME type from data URL if not already set
								if mediaType == "" {
									urlParts := strings.Split(parts[0], ":")
									if len(urlParts) > 1 {
										mimeAndEncoding := strings.Split(urlParts[1], ";")
										if len(mimeAndEncoding) > 0 {
											mediaType = mimeAndEncoding[0]
										}
									}
								}
							}
						}

						// Decode base64 content
						fileData, err := base64.StdEncoding.DecodeString(content)
						if err != nil {
							// If decoding fails, try using the content directly
							fileData = []byte(content)
						}

						// Route to DocumentBlock for document MIME types, ImageBlock for images.
						// Strip any MIME parameters (e.g., ; charset=utf-8), normalize case and whitespace.
						baseMediaType := strings.ToLower(strings.TrimSpace(strings.Split(mediaType, ";")[0]))

						var docFormat types.DocumentFormat
						switch baseMediaType {
						case "application/pdf":
							docFormat = types.DocumentFormatPdf
						case "text/html":
							docFormat = types.DocumentFormatHtml
						case "text/plain":
							docFormat = types.DocumentFormatTxt
						case "text/markdown":
							docFormat = types.DocumentFormatMd
						case "text/csv":
							docFormat = types.DocumentFormatCsv
						case "application/msword":
							docFormat = types.DocumentFormatDoc
						case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
							docFormat = types.DocumentFormatDocx
						case "application/vnd.ms-excel":
							docFormat = types.DocumentFormatXls
						case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
							docFormat = types.DocumentFormatXlsx
						}

						if docFormat != "" {
							contentBlocks = append(contentBlocks, &types.ContentBlockMemberDocument{
								Value: types.DocumentBlock{
									Format: docFormat,
									Name:   aws.String("document"),
									Source: &types.DocumentSourceMemberBytes{
										Value: fileData,
									},
								},
							})
						} else {
							// Treat as image — default to PNG for unknown image types
							var format types.ImageFormat
							switch baseMediaType {
							case "image/png":
								format = types.ImageFormatPng
							case "image/jpeg", "image/jpg":
								format = types.ImageFormatJpeg
							case "image/gif":
								format = types.ImageFormatGif
							case "image/webp":
								format = types.ImageFormatWebp
							default:
								format = types.ImageFormatPng
							}
							contentBlocks = append(contentBlocks, &types.ContentBlockMemberImage{
								Value: types.ImageBlock{
									Format: format,
									Source: &types.ImageSourceMemberBytes{
										Value: fileData,
									},
								},
							})
						}
					} else if part.IsToolRequest() {
						// Handle tool request parts - convert to Bedrock ToolUse blocks
						toolReq := part.ToolRequest
						if toolReq != nil {
							// Create input document from tool request input
							inputDoc := document.NewLazyDocument(toolReq.Input)

							toolUseBlock := &types.ContentBlockMemberToolUse{
								Value: types.ToolUseBlock{
									ToolUseId: aws.String(toolReq.Ref),
									Name:      aws.String(toolReq.Name),
									Input:     inputDoc,
								},
							}
							contentBlocks = append(contentBlocks, toolUseBlock)
						}
					} else if part.IsToolResponse() {
						// Handle tool response parts - convert to Bedrock ToolResult blocks
						toolResp := part.ToolResponse
						if toolResp != nil {
							// Create content for tool result
							var toolResultContent []types.ToolResultContentBlock

							// Convert the output to text content
							if toolResp.Output != nil {
								outputText := ""
								switch output := toolResp.Output.(type) {
								case string:
									outputText = output
								default:
									// Marshal to JSON if not a string
									if jsonBytes, err := json.Marshal(output); err == nil {
										outputText = string(jsonBytes)
									} else {
										outputText = fmt.Sprintf("%v", output)
									}
								}

								toolResultContent = append(toolResultContent, &types.ToolResultContentBlockMemberText{
									Value: outputText,
								})
							}

							toolResultBlock := &types.ContentBlockMemberToolResult{
								Value: types.ToolResultBlock{
									ToolUseId: aws.String(toolResp.Ref),
									Content:   toolResultContent,
									Status:    types.ToolResultStatusSuccess,
								},
							}

							contentBlocks = append(contentBlocks, toolResultBlock)
						}
					} else if part.IsCustom() {
						// Handle custom parts, the plugin currently supports NewCachePointPart
						if cpt, ok := CachePointType(part); ok {
							contentBlocks = append(contentBlocks, &types.ContentBlockMemberCachePoint{
								Value: types.CachePointBlock{
									Type: cpt,
								},
							})
						}
					} else if part.Kind == ai.PartReasoning {
						// Round-trip Bedrock reasoning (thinking) content so the
						// signed text and any redacted block survive into the
						// follow-up request. Reasoning parts without Bedrock
						// metadata produce no blocks (see reasoningPartToContentBlocks).
						contentBlocks = append(contentBlocks, reasoningPartToContentBlocks(part)...)
					}
				}

				bedrockRole := "user"
				if msg.Role == ai.RoleModel {
					bedrockRole = "assistant"
				}

				if len(contentBlocks) > 0 {
					messages = append(messages, types.Message{
						Role:    types.ConversationRole(bedrockRole),
						Content: contentBlocks,
					})
				}
			}
		}

		converseInput.Messages = messages

		// When using tools, AWS Bedrock requires that the conversation doesn't end with an assistant message
		if len(input.Tools) > 0 && len(messages) > 0 {
			lastMessage := messages[len(messages)-1]
			if lastMessage.Role == types.ConversationRoleAssistant {
				// Remove the last assistant message or convert it to user context
				// For now, we'll just remove it to avoid the validation error
				messages = messages[:len(messages)-1]
				converseInput.Messages = messages
			}
		}

		if len(systemPrompts) > 0 {
			converseInput.System = systemPrompts
		}
	}

	// Set inference configuration and any model-specific request fields.
	cfg, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}
	if cfg != nil {
		if inferenceConfig := buildInferenceConfig(cfg); inferenceConfig != nil {
			converseInput.InferenceConfig = inferenceConfig
		}
		if len(cfg.AdditionalModelRequestFields) > 0 {
			converseInput.AdditionalModelRequestFields = document.NewLazyDocument(cfg.AdditionalModelRequestFields)
		}
	}

	// Handle tools
	if len(input.Tools) > 0 {
		var tools []types.Tool
		for _, tool := range input.Tools {
			toolSpec := &types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String(tool.Name),
					Description: aws.String(tool.Description),
				},
			}

			// Convert JSON schema to Bedrock format
			if tool.InputSchema != nil {
				schema, err := b.convertJSONSchemaToBedrockSchema(tool.InputSchema)
				if err == nil && schema != nil {
					toolSpec.Value.InputSchema = *schema
				}
				// If schema conversion fails, tool will still work without detailed schema
			}

			tools = append(tools, toolSpec)
		}

		converseInput.ToolConfig = &types.ToolConfiguration{
			Tools: tools,
		}
	}

	return converseInput, nil
}

// generateTextSync handles synchronous text generation
func (b *Bedrock) generateTextSync(ctx context.Context, input *bedrockruntime.ConverseInput, originalInput *ai.ModelRequest) (*ai.ModelResponse, error) {
	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	// Call Bedrock Converse API
	response, err := b.client.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse failed: %w", err)
	}

	// Convert response to Genkit format
	return b.convertResponse(response, originalInput), nil
}

func (b *Bedrock) convertResponse(response *bedrockruntime.ConverseOutput, originalInput *ai.ModelRequest) *ai.ModelResponse {
	// Initialize response
	modelResponse := &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{},
		},
		FinishReason: ai.FinishReasonStop,
	}

	// Extract output message
	if response.Output != nil {
		if msgMember, ok := response.Output.(*types.ConverseOutputMemberMessage); ok {
			message := msgMember.Value

			// Convert content blocks
			for _, contentBlock := range message.Content {
				switch block := contentBlock.(type) {
				case *types.ContentBlockMemberText:
					modelResponse.Message.Content = append(modelResponse.Message.Content,
						ai.NewTextPart(block.Value))

				case *types.ContentBlockMemberToolUse:
					// Handle tool use blocks - convert to proper Genkit tool request
					toolUse := block.Value

					// Extract tool input from the AWS document format
					var toolInput interface{}
					if toolUse.Input != nil {
						// Unmarshal the tool input document to a map
						var inputMap map[string]interface{}
						if err := toolUse.Input.UnmarshalSmithyDocument(&inputMap); err == nil {
							// Convert tool input based on the original tool schema
							toolInput = b.convertToolInputTypes(inputMap, aws.ToString(toolUse.Name), originalInput.Tools)
						} else {
							// Fallback: create empty map for failed unmarshaling
							toolInput = map[string]interface{}{
								"_unmarshal_error": err.Error(),
								"_tool_use_id":     aws.ToString(toolUse.ToolUseId),
							}
						}
					} else {
						toolInput = map[string]interface{}{}
					}

					// Create a proper tool request part
					toolRequest := &ai.ToolRequest{
						Name:  aws.ToString(toolUse.Name),
						Input: toolInput,
						Ref:   aws.ToString(toolUse.ToolUseId),
					}

					modelResponse.Message.Content = append(modelResponse.Message.Content,
						ai.NewToolRequestPart(toolRequest))

				case *types.ContentBlockMemberReasoningContent:
					// Reasoning ("thinking") content: carry the signed text and
					// any redacted block so it can be replayed on the next turn.
					if part, err := reasoningBlockToPart(block.Value); err == nil && part != nil {
						modelResponse.Message.Content = append(modelResponse.Message.Content, part)
					}
				}
			}
		}
	}

	// Convert finish reason
	modelResponse.FinishReason = convertStopReasonToGenkit(response.StopReason)

	// Extract usage information (if available in the API)
	if response.Usage != nil {
		// Map AWS Bedrock TokenUsage to Genkit GenerationUsage
		modelResponse.Usage = &ai.GenerationUsage{
			InputTokens:         int(aws.ToInt32(response.Usage.InputTokens)),
			OutputTokens:        int(aws.ToInt32(response.Usage.OutputTokens)),
			TotalTokens:         int(aws.ToInt32(response.Usage.TotalTokens)),
			CachedContentTokens: int(aws.ToInt32(response.Usage.CacheReadInputTokens)),
		}
	}

	// If no content was extracted, add placeholder
	if len(modelResponse.Message.Content) == 0 {
		modelResponse.Message.Content = append(modelResponse.Message.Content,
			ai.NewTextPart(""))
	}

	return modelResponse
}

// configFromRequest decodes input.Config into a *Config. It accepts the typed
// *Config/Config, *ai.GenerationCommonConfig/ai.GenerationCommonConfig, and the
// historical map[string]any shape (used on resumed/serialized flows). It
// returns (nil, nil) when no config is provided.
func configFromRequest(input *ai.ModelRequest) (*Config, error) {
	if input == nil || input.Config == nil {
		return nil, nil
	}
	switch v := input.Config.(type) {
	case *Config:
		return v, nil
	case Config:
		return &v, nil
	case *ai.GenerationCommonConfig:
		return configFromGenerationCommonConfig(v), nil
	case ai.GenerationCommonConfig:
		return configFromGenerationCommonConfig(&v), nil
	case map[string]interface{}:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("bedrock: marshal config: %w", err)
		}
		var c Config
		if err := json.Unmarshal(b, &c); err != nil {
			return nil, fmt.Errorf("bedrock: decode config: %w", err)
		}
		// Preserve the historical max-token keys, which differ from Config's
		// json tag ("maxTokens"). The map values may decode as float64 (JSON)
		// or int (a directly-constructed Go map).
		if c.MaxTokens == 0 {
			if mt, ok := mapInt(v, "maxOutputTokens"); ok {
				c.MaxTokens = mt
			} else if mt, ok := mapInt(v, "max_tokens"); ok {
				c.MaxTokens = mt
			}
		}
		return &c, nil
	default:
		return nil, fmt.Errorf("bedrock: unexpected config type %T, want *bedrock.Config, *ai.GenerationCommonConfig, or map[string]any", input.Config)
	}
}

// mapInt reads an integer-valued key from a config map, tolerating the float64
// that JSON-decoded numbers arrive as alongside a plain int.
func mapInt(m map[string]interface{}, key string) (int, bool) {
	switch v := m[key].(type) {
	case int:
		return v, true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	default:
		return 0, false
	}
}

func configFromGenerationCommonConfig(v *ai.GenerationCommonConfig) *Config {
	if v == nil {
		return nil
	}
	cfg := &Config{
		MaxTokens:     v.MaxOutputTokens,
		StopSequences: v.StopSequences,
	}
	if v.Temperature != 0 {
		t := float32(v.Temperature)
		cfg.Temperature = &t
	}
	if v.TopP != 0 {
		p := float32(v.TopP)
		cfg.TopP = &p
	}
	return cfg
}

// buildInferenceConfig maps a *Config onto Bedrock's InferenceConfiguration. It
// returns nil when nothing is set, leaving Bedrock to apply its own defaults
// (MaxTokens is only sent when explicitly provided).
func buildInferenceConfig(cfg *Config) *types.InferenceConfiguration {
	if cfg == nil {
		return nil
	}
	ic := &types.InferenceConfiguration{}
	set := false
	if cfg.MaxTokens > 0 {
		ic.MaxTokens = aws.Int32(int32(cfg.MaxTokens))
		set = true
	}
	if cfg.Temperature != nil {
		ic.Temperature = cfg.Temperature
		set = true
	}
	if cfg.TopP != nil {
		ic.TopP = cfg.TopP
		set = true
	}
	if len(cfg.StopSequences) > 0 {
		ic.StopSequences = cfg.StopSequences
		set = true
	}
	if !set {
		return nil
	}
	return ic
}

// reasoningPartToContentBlocks converts a reasoning ai.Part back into Bedrock
// reasoning content blocks. Only Bedrock-originated reasoning (carrying the
// signature and/or redacted metadata) is emitted; a generic reasoning part
// produces no blocks so it cannot corrupt the follow-up request.
func reasoningPartToContentBlocks(p *ai.Part) []types.ContentBlock {
	var blocks []types.ContentBlock
	if redacted := metadataBytes(p.Metadata, redactedReasoningMetadataKey); len(redacted) > 0 {
		blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberRedactedContent{Value: redacted},
		})
	}
	if signature := metadataBytes(p.Metadata, reasoningSignatureMetadataKey); p.Text != "" && len(signature) > 0 {
		blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberReasoningText{
				Value: types.ReasoningTextBlock{
					Text:      aws.String(p.Text),
					Signature: aws.String(string(signature)),
				},
			},
		})
	}
	return blocks
}

// reasoningBlockToPart converts a Bedrock reasoning content block into an ai
// reasoning Part, or (nil, nil) when the block is empty.
func reasoningBlockToPart(block types.ReasoningContentBlock) (*ai.Part, error) {
	switch rc := block.(type) {
	case *types.ReasoningContentBlockMemberReasoningText:
		if rc.Value.Text == nil && rc.Value.Signature == nil {
			return nil, nil
		}
		return newBedrockReasoningPart(aws.ToString(rc.Value.Text), aws.ToString(rc.Value.Signature), nil), nil
	case *types.ReasoningContentBlockMemberRedactedContent:
		if len(rc.Value) == 0 {
			return nil, nil
		}
		return newBedrockReasoningPart("", "", rc.Value), nil
	default:
		return nil, fmt.Errorf("bedrock: unhandled reasoning content variant %T", block)
	}
}

// convertToolInputTypes converts tool input parameters to the correct types based on the tool schema
func (b *Bedrock) convertToolInputTypes(inputMap map[string]interface{}, toolName string, tools []*ai.ToolDefinition) interface{} {
	// Find the tool definition for this tool call
	var targetTool *ai.ToolDefinition
	for _, tool := range tools {
		if tool.Name == toolName {
			targetTool = tool
			break
		}
	}

	// If we can't find the tool definition, return the original input
	if targetTool == nil || targetTool.InputSchema == nil {
		return inputMap
	}

	// Convert the input map based on the schema
	return b.convertMapWithSchema(inputMap, targetTool.InputSchema)
}

// convertMapWithSchema recursively converts a map's values to match the expected schema types
func (b *Bedrock) convertMapWithSchema(inputMap map[string]interface{}, schema map[string]any) interface{} {
	if schema == nil {
		return inputMap
	}

	result := make(map[string]interface{})

	// Handle object schema with properties
	if schemaType, ok := schema["type"].(string); ok && schemaType == "object" {
		if properties, ok := schema["properties"].(map[string]any); ok {
			for key, value := range inputMap {
				if propSchema, exists := properties[key]; exists {
					if propSchemaMap, ok := propSchema.(map[string]any); ok {
						result[key] = b.convertValueWithSchema(value, propSchemaMap)
					} else {
						result[key] = value
					}
				} else {
					result[key] = value // Keep original value if no schema
				}
			}
			return result
		}
	}

	// For non-object schemas, convert the whole map as-is
	return inputMap
}

// convertValueWithSchema converts a single value to match the expected schema type
func (b *Bedrock) convertValueWithSchema(value interface{}, schema map[string]any) interface{} {
	if schema == nil {
		return value
	}

	schemaType, hasType := schema["type"].(string)
	if !hasType {
		return value
	}

	// Handle AWS document.Number type specifically
	if docNum, ok := value.(smithydoc.Number); ok {
		switch schemaType {
		case "number":
			if floatVal, err := docNum.Float64(); err == nil {
				return floatVal
			}
		case "integer":
			if intVal, err := docNum.Int64(); err == nil {
				return intVal
			}
		}
	}

	// Handle string values that need to be converted to numbers
	if strValue, ok := value.(string); ok {
		switch schemaType {
		case "number", "integer":
			// Try to convert string to number
			if floatVal, err := strconv.ParseFloat(strValue, 64); err == nil {
				if schemaType == "integer" {
					return int64(floatVal)
				}
				return floatVal
			}
		case "boolean":
			// Try to convert string to boolean
			if boolVal, err := strconv.ParseBool(strValue); err == nil {
				return boolVal
			}
		}
	}

	// Handle numeric types that need conversion
	switch schemaType {
	case "number":
		switch v := value.(type) {
		case int:
			return float64(v)
		case int32:
			return float64(v)
		case int64:
			return float64(v)
		case float32:
			return float64(v)
		case float64:
			return v
		}
	case "integer":
		switch v := value.(type) {
		case int:
			return int64(v)
		case int32:
			return int64(v)
		case int64:
			return v
		case float32:
			return int64(v)
		case float64:
			return int64(v)
		}
	}

	// Handle arrays
	if schemaType == "array" {
		if items, ok := schema["items"].(map[string]any); ok {
			if arrayValue, ok := value.([]interface{}); ok {
				result := make([]interface{}, len(arrayValue))
				for i, item := range arrayValue {
					result[i] = b.convertValueWithSchema(item, items)
				}
				return result
			}
		}
	}

	// Handle objects
	if schemaType == "object" {
		if mapValue, ok := value.(map[string]interface{}); ok {
			return b.convertMapWithSchema(mapValue, schema)
		}
	}

	// Return original value if no conversion needed
	return value
}

// convertJSONSchemaToBedrockSchema converts a JSON schema to Bedrock ToolInputSchema format
func (b *Bedrock) convertJSONSchemaToBedrockSchema(schema any) (*types.ToolInputSchema, error) {
	if schema == nil {
		return nil, fmt.Errorf("schema is nil")
	}

	// Convert schema to a map[string]interface{} format
	schemaMap, err := b.normalizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize schema: %w", err)
	}

	// Create a document using the AWS SDK's NewLazyDocument function
	doc := document.NewLazyDocument(schemaMap)

	// Create the JSON schema member
	jsonSchemaMember := &types.ToolInputSchemaMemberJson{
		Value: doc,
	}

	// Return as ToolInputSchema interface
	var bedrockSchema types.ToolInputSchema = jsonSchemaMember
	return &bedrockSchema, nil
}

// normalizeSchema converts various schema formats to a standard map[string]interface{}
func (b *Bedrock) normalizeSchema(schema any) (map[string]interface{}, error) {
	switch s := schema.(type) {
	case map[string]interface{}:
		// Already in the correct format - validate it's a proper JSON Schema
		return b.validateAndNormalizeJSONSchema(s), nil
	case string:
		// Try to parse JSON string
		var schemaMap map[string]interface{}
		if err := json.Unmarshal([]byte(s), &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON: %w", err)
		}
		return b.validateAndNormalizeJSONSchema(schemaMap), nil
	case []byte:
		// Try to parse JSON bytes
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(s, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON bytes: %w", err)
		}
		return b.validateAndNormalizeJSONSchema(schemaMap), nil
	default:
		// Try to marshal and unmarshal to get a map
		jsonData, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(jsonData, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
		}
		return b.validateAndNormalizeJSONSchema(schemaMap), nil
	}
}

// validateAndNormalizeJSONSchema ensures the schema is a valid JSON Schema and adds required fields
func (b *Bedrock) validateAndNormalizeJSONSchema(schema map[string]interface{}) map[string]interface{} {
	// Make a copy to avoid modifying the original
	normalized := make(map[string]interface{})
	for k, v := range schema {
		normalized[k] = v
	}

	// Ensure we have a type field - default to "object" if not specified
	if _, exists := normalized["type"]; !exists {
		normalized["type"] = "object"
	}

	// Ensure we have a properties field for object types
	if normalized["type"] == "object" {
		if _, exists := normalized["properties"]; !exists {
			normalized["properties"] = map[string]interface{}{}
		}
	}

	// Add JSON Schema version if not present
	if _, exists := normalized["$schema"]; !exists {
		normalized["$schema"] = "http://json-schema.org/draft-07/schema#"
	}

	return normalized
}

// Helper functions for creating JSON Schema patterns

// NewObjectSchema creates a JSON Schema for an object with the specified properties
func NewObjectSchema(properties map[string]interface{}, required []string) map[string]interface{} {
	schema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// NewStringSchema creates a JSON Schema for a string with optional constraints
func NewStringSchema(description string, enum []string) map[string]interface{} {
	schema := map[string]interface{}{
		"type": "string",
	}

	if description != "" {
		schema["description"] = description
	}

	if len(enum) > 0 {
		schema["enum"] = enum
	}

	return schema
}

// NewNumberSchema creates a JSON Schema for a number with optional constraints
func NewNumberSchema(description string, minimum, maximum *float64) map[string]interface{} {
	schema := map[string]interface{}{
		"type": "number",
	}

	if description != "" {
		schema["description"] = description
	}

	if minimum != nil {
		schema["minimum"] = *minimum
	}

	if maximum != nil {
		schema["maximum"] = *maximum
	}

	return schema
}

// NewArraySchema creates a JSON Schema for an array with the specified item type
func NewArraySchema(itemSchema map[string]interface{}, description string) map[string]interface{} {
	schema := map[string]interface{}{
		"type":  "array",
		"items": itemSchema,
	}

	if description != "" {
		schema["description"] = description
	}

	return schema
}

// Helper functions

// convertStopReasonToGenkit converts Bedrock stop reason to Genkit finish reason
func convertStopReasonToGenkit(stopReason types.StopReason) ai.FinishReason {
	switch stopReason {
	case types.StopReasonEndTurn:
		return ai.FinishReasonStop
	case types.StopReasonMaxTokens:
		return ai.FinishReasonLength
	case types.StopReasonStopSequence:
		return ai.FinishReasonStop
	case types.StopReasonToolUse:
		return ai.FinishReasonStop
	case types.StopReasonContentFiltered:
		return ai.FinishReasonBlocked
	default:
		return ai.FinishReasonOther
	}
}
