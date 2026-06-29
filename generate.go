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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
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

const (
	defaultClaudeMaxTokens         int32 = 4096
	defaultExtendedClaudeMaxTokens int32 = 8192
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

	cfg, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}

	systemPrompts, messages, err := convertMessages(input.Messages)
	if err != nil {
		return nil, err
	}

	// When using tools, AWS Bedrock requires that the conversation doesn't end
	// with an assistant message.
	if len(input.Tools) > 0 && len(messages) > 0 {
		if messages[len(messages)-1].Role == types.ConversationRoleAssistant {
			messages = messages[:len(messages)-1]
		}
	}

	inferenceConfig := buildInferenceConfig(cfg)
	if maxTokens, ok := defaultMaxTokensForModel(modelName); ok {
		if inferenceConfig == nil {
			inferenceConfig = &types.InferenceConfiguration{}
		}
		if inferenceConfig.MaxTokens == nil {
			inferenceConfig.MaxTokens = aws.Int32(maxTokens)
		}
	}

	converseInput := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(modelName),
		Messages:        messages,
		System:          systemPrompts,
		InferenceConfig: inferenceConfig,
	}

	if cfg != nil {
		if len(cfg.AdditionalModelRequestFields) > 0 {
			converseInput.AdditionalModelRequestFields = document.NewLazyDocument(cfg.AdditionalModelRequestFields)
		}
	}

	// Handle tools
	if len(input.Tools) > 0 {
		if cfg != nil && cfg.ToolChoice == ToolChoiceNone {
			return converseInput, nil
		}
		tools, err := b.convertTools(input.Tools)
		if err != nil {
			return nil, err
		}
		converseInput.ToolConfig = &types.ToolConfiguration{Tools: tools}
		if cfg != nil && cfg.ToolChoice != "" {
			choice, err := convertToolChoice(cfg.ToolChoice, input.Tools)
			if err != nil {
				return nil, err
			}
			converseInput.ToolConfig.ToolChoice = choice
		}
	}

	return converseInput, nil
}

// convertMessages walks the ai.ModelRequest messages and produces a system
// block list plus the user/assistant/tool conversation.
func convertMessages(msgs []*ai.Message) ([]types.SystemContentBlock, []types.Message, error) {
	var system []types.SystemContentBlock
	var messages []types.Message
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		if msg.Role == ai.RoleSystem {
			for _, part := range msg.Content {
				if part == nil {
					continue
				}
				if cpt, ok := CachePointType(part); ok {
					system = append(system, &types.SystemContentBlockMemberCachePoint{
						Value: types.CachePointBlock{Type: cpt},
					})
					continue
				}
				if part.IsText() {
					system = append(system, &types.SystemContentBlockMemberText{Value: part.Text})
				}
			}
			continue
		}

		role, err := convertRole(msg.Role)
		if err != nil {
			return nil, nil, err
		}
		blocks, err := partsToContentBlocks(msg.Content)
		if err != nil {
			return nil, nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		messages = append(messages, types.Message{Role: role, Content: blocks})
	}
	return system, messages, nil
}

func convertRole(role ai.Role) (types.ConversationRole, error) {
	switch role {
	case ai.RoleUser, ai.RoleTool:
		return types.ConversationRoleUser, nil
	case ai.RoleModel:
		return types.ConversationRoleAssistant, nil
	default:
		return "", fmt.Errorf("bedrock: unsupported role %q", role)
	}
}

func partsToContentBlocks(parts []*ai.Part) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock
	for _, part := range parts {
		if part == nil {
			continue
		}
		switch {
		case part.IsText():
			blocks = append(blocks, &types.ContentBlockMemberText{Value: part.Text})
		case part.IsMedia():
			block, err := mediaToBlock(part)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case part.IsToolRequest():
			toolReq := part.ToolRequest
			if toolReq == nil {
				continue
			}
			blocks = append(blocks, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: aws.String(toolReq.Ref),
					Name:      aws.String(toolReq.Name),
					Input:     document.NewLazyDocument(toolReq.Input),
				},
			})
		case part.IsToolResponse():
			toolResp := part.ToolResponse
			if toolResp == nil {
				continue
			}
			outputText, err := toolResponseText(toolResp.Output)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, &types.ContentBlockMemberToolResult{
				Value: types.ToolResultBlock{
					ToolUseId: aws.String(toolResp.Ref),
					Content: []types.ToolResultContentBlock{
						&types.ToolResultContentBlockMemberText{Value: outputText},
					},
					Status: types.ToolResultStatusSuccess,
				},
			})
		case part.IsCustom():
			if cpt, ok := CachePointType(part); ok {
				blocks = append(blocks, &types.ContentBlockMemberCachePoint{
					Value: types.CachePointBlock{Type: cpt},
				})
			}
		case part.Kind == ai.PartReasoning:
			blocks = append(blocks, reasoningPartToContentBlocks(part)...)
		}
	}
	return blocks, nil
}

func toolResponseText(output any) (string, error) {
	if output == nil {
		return "", nil
	}
	if s, ok := output.(string); ok {
		return s, nil
	}
	jsonBytes, err := json.Marshal(output)
	if err != nil {
		return "", fmt.Errorf("bedrock: marshal tool response: %w", err)
	}
	return string(jsonBytes), nil
}

func mediaToBlock(part *ai.Part) (types.ContentBlock, error) {
	mime := mediaMIME(part)
	if mime == "" {
		return nil, errors.New("bedrock: media part has no content type")
	}
	fileData, err := decodeMediaPayload(part.Text)
	if err != nil {
		return nil, err
	}
	if format := documentFormatFor(mime); format != "" {
		return &types.ContentBlockMemberDocument{
			Value: types.DocumentBlock{
				Format: format,
				Name:   aws.String("document"),
				Source: &types.DocumentSourceMemberBytes{Value: fileData},
			},
		}, nil
	}
	if format := imageFormatFor(mime); format != "" {
		return &types.ContentBlockMemberImage{
			Value: types.ImageBlock{
				Format: format,
				Source: &types.ImageSourceMemberBytes{Value: fileData},
			},
		}, nil
	}
	return nil, fmt.Errorf("bedrock: unsupported media MIME type %q (must be png/jpeg/gif/webp or one of pdf/csv/doc/docx/xls/xlsx/html/txt/md)", mime)
}

func mediaMIME(part *ai.Part) string {
	mime := strings.TrimSpace(part.ContentType)
	if mime == "" && strings.HasPrefix(part.Text, "data:") {
		header, _, ok := strings.Cut(part.Text, ",")
		if ok {
			header = strings.TrimPrefix(header, "data:")
			mime, _, _ = strings.Cut(header, ";")
		}
	}
	mime, _, _ = strings.Cut(mime, ";")
	return strings.ToLower(strings.TrimSpace(mime))
}

// decodeMediaPayload accepts either a raw "data:<mime>;base64,..." URL or a
// bare base64 string and returns decoded bytes. Bedrock expects raw bytes; the
// SDK base64-encodes them for the wire.
func decodeMediaPayload(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, errors.New("bedrock: media part has empty data")
	}
	if i := strings.Index(s, ";base64,"); i >= 0 {
		s = strings.TrimSpace(s[i+len(";base64,"):])
	} else if strings.HasPrefix(s, "data:") {
		return nil, errors.New("bedrock: data URL must be base64-encoded (use ';base64,' prefix)")
	} else if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
		return nil, errors.New("bedrock: remote URLs are not supported; use a data URL or base64-encoded data")
	}
	fileData, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("bedrock: decode base64 media: %w", err)
	}
	return fileData, nil
}

func imageFormatFor(mime string) types.ImageFormat {
	switch mime {
	case "image/png":
		return types.ImageFormatPng
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg
	case "image/gif":
		return types.ImageFormatGif
	case "image/webp":
		return types.ImageFormatWebp
	default:
		return ""
	}
}

func documentFormatFor(mime string) types.DocumentFormat {
	switch mime {
	case "application/pdf":
		return types.DocumentFormatPdf
	case "text/html":
		return types.DocumentFormatHtml
	case "text/plain":
		return types.DocumentFormatTxt
	case "text/markdown":
		return types.DocumentFormatMd
	case "text/csv":
		return types.DocumentFormatCsv
	case "application/msword":
		return types.DocumentFormatDoc
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return types.DocumentFormatDocx
	case "application/vnd.ms-excel":
		return types.DocumentFormatXls
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return types.DocumentFormatXlsx
	default:
		return ""
	}
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
	modelResponse, err := b.convertResponse(response, originalInput)
	if err != nil {
		return nil, err
	}
	modelResponse.Request = originalInput
	return modelResponse, nil
}

func (b *Bedrock) convertResponse(response *bedrockruntime.ConverseOutput, originalInput *ai.ModelRequest) (*ai.ModelResponse, error) {
	if response == nil {
		return nil, errors.New("bedrock: converse response is nil")
	}

	var parts []*ai.Part
	if response.Output != nil {
		msgMember, ok := response.Output.(*types.ConverseOutputMemberMessage)
		if !ok {
			return nil, fmt.Errorf("bedrock: unexpected output variant %T", response.Output)
		}
		var err error
		parts, err = b.contentBlocksToParts(msgMember.Value.Content, originalInput)
		if err != nil {
			return nil, err
		}
	}
	if len(parts) == 0 {
		parts = append(parts, ai.NewTextPart(""))
	}
	return &ai.ModelResponse{
		Message:      &ai.Message{Role: ai.RoleModel, Content: parts},
		FinishReason: convertStopReasonToGenkit(response.StopReason),
		Usage:        usageFromTokens(response.Usage),
		Request:      originalInput,
	}, nil
}

func (b *Bedrock) contentBlocksToParts(blocks []types.ContentBlock, originalInput *ai.ModelRequest) ([]*ai.Part, error) {
	out := make([]*ai.Part, 0, len(blocks))
	for _, contentBlock := range blocks {
		switch block := contentBlock.(type) {
		case *types.ContentBlockMemberText:
			out = append(out, ai.NewTextPart(block.Value))
		case *types.ContentBlockMemberToolUse:
			toolUse := block.Value
			toolInput, err := b.unwrapToolInput(toolUse.Input, aws.ToString(toolUse.Name), originalInput)
			if err != nil {
				return nil, err
			}
			out = append(out, ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  aws.ToString(toolUse.Name),
				Input: toolInput,
				Ref:   aws.ToString(toolUse.ToolUseId),
			}))
		case *types.ContentBlockMemberReasoningContent:
			part, err := reasoningBlockToPart(block.Value)
			if err != nil {
				return nil, err
			}
			if part != nil {
				out = append(out, part)
			}
		default:
			return nil, fmt.Errorf("bedrock: unhandled response content variant %T", contentBlock)
		}
	}
	return out, nil
}

func (b *Bedrock) unwrapToolInput(input document.Interface, toolName string, originalInput *ai.ModelRequest) (any, error) {
	if input == nil {
		return map[string]any{}, nil
	}
	var decoded any
	data, err := input.MarshalSmithyDocument()
	if err != nil {
		return nil, fmt.Errorf("bedrock: decode tool input: %w", err)
	}
	if len(data) == 0 {
		return nil, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		return nil, fmt.Errorf("bedrock: decode tool input: %w", err)
	}
	var tools []*ai.ToolDefinition
	if originalInput != nil {
		tools = originalInput.Tools
	}
	if inputMap, ok := decoded.(map[string]any); ok {
		return b.convertToolInputTypes(inputMap, toolName, tools), nil
	}
	return decoded, nil
}

func usageFromTokens(usage *types.TokenUsage) *ai.GenerationUsage {
	if usage == nil {
		return nil
	}
	return &ai.GenerationUsage{
		InputTokens:         int(aws.ToInt32(usage.InputTokens)),
		OutputTokens:        int(aws.ToInt32(usage.OutputTokens)),
		TotalTokens:         int(aws.ToInt32(usage.TotalTokens)),
		CachedContentTokens: int(aws.ToInt32(usage.CacheReadInputTokens)),
	}
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

func defaultMaxTokensForModel(modelName string) (int32, bool) {
	name := strings.ToLower(modelName)
	if !strings.Contains(name, "claude") {
		return 0, false
	}
	if strings.Contains(name, "claude-3-5") ||
		strings.Contains(name, "claude-3-7") ||
		strings.Contains(name, "claude-4") ||
		strings.Contains(name, "claude-haiku-4") ||
		strings.Contains(name, "claude-sonnet-4") ||
		strings.Contains(name, "claude-opus-4") {
		return defaultExtendedClaudeMaxTokens, true
	}
	return defaultClaudeMaxTokens, true
}

// buildInferenceConfig maps a *Config onto Bedrock's InferenceConfiguration.
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

func (b *Bedrock) convertTools(tools []*ai.ToolDefinition) ([]types.Tool, error) {
	out := make([]types.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool == nil {
			return nil, errors.New("bedrock: tool definition required")
		}
		if tool.Name == "" {
			return nil, errors.New("bedrock: tool name required")
		}

		schema := tool.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		var inputSchema types.ToolInputSchema
		if bedrockSchema, err := b.convertJSONSchemaToBedrockSchema(schema); err == nil && bedrockSchema != nil {
			inputSchema = *bedrockSchema
		}

		out = append(out, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        aws.String(tool.Name),
				Description: aws.String(tool.Description),
				InputSchema: inputSchema,
			},
		})
	}
	return out, nil
}

func convertToolChoice(choice string, tools []*ai.ToolDefinition) (types.ToolChoice, error) {
	switch choice {
	case "", ToolChoiceAuto:
		return &types.ToolChoiceMemberAuto{}, nil
	case ToolChoiceRequired, ToolChoiceAny:
		return &types.ToolChoiceMemberAny{}, nil
	default:
		for _, tool := range tools {
			if tool != nil && tool.Name == choice {
				return &types.ToolChoiceMemberTool{
					Value: types.SpecificToolChoice{Name: aws.String(choice)},
				}, nil
			}
		}
		return nil, fmt.Errorf("bedrock: ToolChoice %q does not match any declared tool", choice)
	}
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

	if num, ok := value.(json.Number); ok {
		switch schemaType {
		case "number":
			if floatVal, err := num.Float64(); err == nil {
				return floatVal
			}
		case "integer":
			if intVal, err := num.Int64(); err == nil {
				return intVal
			}
			if floatVal, err := num.Float64(); err == nil {
				return int64(floatVal)
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
	if _, err := json.Marshal(schemaMap); err != nil {
		return nil, fmt.Errorf("failed to validate schema JSON: %w", err)
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
	case types.StopReasonEndTurn, types.StopReasonStopSequence, types.StopReasonToolUse:
		return ai.FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return ai.FinishReasonLength
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return ai.FinishReasonBlocked
	case types.StopReasonMalformedModelOutput, types.StopReasonMalformedToolUse:
		return ai.FinishReasonOther
	default:
		return ai.FinishReasonOther
	}
}
