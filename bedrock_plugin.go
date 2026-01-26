// Copyright 2025 Xavier Portilla Edo
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

// Package bedrock provides a comprehensive AWS Bedrock plugin for Genkit Go.
// This plugin supports text generation, image generation, and embedding capabilities
// using AWS Bedrock foundation models via the Converse API.
//
// This implementation follows the same patterns as the existing Genkit plugins:
// - ollama: https://github.com/firebase/genkit/blob/main/go/plugins/ollama/ollama.go
// - gemini: https://github.com/firebase/genkit/blob/main/go/plugins/googlegenai/gemini.go
package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithydoc "github.com/aws/smithy-go/document"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// Type aliases for better readability
type (
	BedrockClient = *bedrockruntime.Client
	Role          = ai.Role
	ToolChoice    = string
	FinishReason  = string
)

// ModelCapability represents the capabilities of a model
type ModelCapability struct {
	Multimodal bool // Supports image/media inputs
	Tools      bool // Supports function calling
}

// Constants
const provider = "bedrock"

// Role constants (would come from ai package)
const (
	RoleUser   Role = "user"
	RoleModel  Role = "assistant"
	RoleSystem Role = "system"
	RoleTool   Role = "tool"
)

// Tool choice constants
const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceRequired ToolChoice = "required"
	ToolChoiceNone     ToolChoice = "none"
)

// Finish reason constants
const (
	FinishReasonStop    FinishReason = "stop"
	FinishReasonLength  FinishReason = "length"
	FinishReasonBlocked FinishReason = "blocked"
	FinishReasonOther   FinishReason = "other"
	FinishReasonUnknown FinishReason = "unknown"
)

const bedrockCachePointTypeKey = "bedrockCachePointType"

// inferenceProfilePrefixes lists all valid inference profile region prefixes.
// When an inference profile is passed instead of a model ID, the prefix is
// stripped before checking capabilities.
var inferenceProfilePrefixes = []string{
	"global.",
	"us.",
	"eu.",
	"jp.",
	"apac.",
	"au.",
	"us-gov.",
}

// modelCapabilities maps model IDs to their capabilities.
// This consolidates the previous multimodalModels and toolSupportedModels lists.
var modelCapabilities = map[string]ModelCapability{
	// Anthropic Claude 3 models
	"anthropic.claude-3-haiku-20240307-v1:0":    {Multimodal: true, Tools: true},
	"anthropic.claude-3-sonnet-20240229-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-3-opus-20240229-v1:0":     {Multimodal: true, Tools: true},
	"anthropic.claude-3-5-haiku-20241022-v1:0":  {Multimodal: false, Tools: true},
	"anthropic.claude-3-5-sonnet-20240620-v1:0": {Multimodal: true, Tools: true},
	"anthropic.claude-3-5-sonnet-20241022-v2:0": {Multimodal: true, Tools: true},
	"anthropic.claude-3-7-sonnet-20250219-v1:0": {Multimodal: true, Tools: true},
	// Anthropic Claude 4/4.5 models
	"anthropic.claude-opus-4-20250514-v1:0":     {Multimodal: true, Tools: true},
	"anthropic.claude-sonnet-4-20250514-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-sonnet-4-5-20250929-v1:0": {Multimodal: true, Tools: true},
	"anthropic.claude-opus-4-5-20251101-v1:0":   {Multimodal: true, Tools: true},
	// Amazon Nova models
	"amazon.nova-micro-v1:0":   {Multimodal: false, Tools: true},
	"amazon.nova-lite-v1:0":    {Multimodal: true, Tools: true},
	"amazon.nova-pro-v1:0":     {Multimodal: true, Tools: true},
	"amazon.nova-premier-v1:0": {Multimodal: true, Tools: true},
	// Cohere Command models
	"cohere.command-r-v1:0":      {Multimodal: false, Tools: true},
	"cohere.command-r-plus-v1:0": {Multimodal: false, Tools: true},
	// Mistral models
	"mistral.mistral-large-2402-v1:0": {Multimodal: false, Tools: true},
	"mistral.mistral-large-2407-v1:0": {Multimodal: false, Tools: true},
	"mistral.mistral-small-2402-v1:0": {Multimodal: false, Tools: true},
	"mistral.pixtral-large-2502-v1:0": {Multimodal: true, Tools: true},
	// AI21 Labs Jamba models
	"ai21.jamba-1-5-large-v1:0": {Multimodal: false, Tools: true},
	"ai21.jamba-1-5-mini-v1:0":  {Multimodal: false, Tools: true},
	// Meta Llama models
	"meta.llama3-8b-instruct-v1:0":           {Multimodal: false, Tools: true},
	"meta.llama3-70b-instruct-v1:0":          {Multimodal: false, Tools: true},
	"meta.llama3-1-8b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-1-70b-instruct-v1:0":        {Multimodal: false, Tools: true},
	"meta.llama3-1-405b-instruct-v1:0":       {Multimodal: false, Tools: true},
	"meta.llama3-2-1b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-2-3b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-2-11b-instruct-v1:0":        {Multimodal: true, Tools: true},
	"meta.llama3-2-90b-instruct-v1:0":        {Multimodal: true, Tools: true},
	"meta.llama3-3-70b-instruct-v1:0":        {Multimodal: false, Tools: true},
	"meta.llama4-maverick-17b-instruct-v1:0": {Multimodal: true, Tools: true},
	"meta.llama4-scout-17b-instruct-v1:0":    {Multimodal: true, Tools: true},
	// DeepSeek models
	"deepseek.r1-v1:0": {Multimodal: false, Tools: true},
	// Writer models
	"writer.palmyra-x4-v1:0": {Multimodal: false, Tools: true},
	"writer.palmyra-x5-v1:0": {Multimodal: false, Tools: true},
	// TwelveLabs models
	"twelvelabs.pegasus-1-2-v1:0": {Multimodal: false, Tools: true},
}

// Bedrock provides configuration options for the AWS Bedrock plugin.
type Bedrock struct {
	Region         string        // AWS region (optional, uses AWS_REGION or us-east-1)
	MaxRetries     int           // Maximum number of retries (default: 3)
	RequestTimeout time.Duration // Request timeout (default: 30s)
	AWSConfig      *aws.Config   // Custom AWS config (optional)

	mu      sync.Mutex // Mutex to control access
	client  BedrockClient
	initted bool // Whether the plugin has been initialized
}

// ModelDefinition represents a model with its name and type.
type ModelDefinition struct {
	Name string // Model ID as used in AWS Bedrock
	Type string // Type: "chat", "text", "image", "embedding"
}

// Name returns the provider name.
func (b *Bedrock) Name() string {
	return provider
}

// Init initializes the AWS Bedrock plugin.
// This method follows the same pattern as the Ollama plugin.
func (b *Bedrock) Init(ctx context.Context) []api.Action {
	b.mu.Lock()

	if b.initted {
		b.mu.Unlock()
		panic("bedrock: Init already called")
	}

	// Set defaults
	if b.Region == "" {
		b.Region = "us-east-1" // Default region
	}
	if b.MaxRetries == 0 {
		b.MaxRetries = 3
	}
	if b.RequestTimeout == 0 {
		b.RequestTimeout = 30 * time.Second
	}

	// Load AWS configuration
	var awsConfig aws.Config
	var err error

	if b.AWSConfig != nil {
		awsConfig = *b.AWSConfig
	} else {
		// Load default AWS configuration
		awsConfig, err = config.LoadDefaultConfig(ctx,
			config.WithRegion(b.Region),
			config.WithRetryMaxAttempts(b.MaxRetries),
		)
		if err != nil {
			panic(fmt.Sprintf("bedrock: failed to load AWS config: %v", err))
		}
	}

	// Create Bedrock Runtime client
	b.client = bedrockruntime.NewFromConfig(awsConfig)

	b.initted = true

	// Release the mutex
	b.mu.Unlock()

	// Don't defer unlock since we already unlocked manually
	return []api.Action{}
}

// DefineModel defines a model in the registry.
// This follows the same pattern as the Anthropic plugin's DefineModel method.
func (b *Bedrock) DefineModel(g *genkit.Genkit, model ModelDefinition, info *ai.ModelInfo) ai.Model {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.initted {
		panic("bedrock: Init not called")
	}

	// Auto-detect model capabilities if not provided
	if info == nil {
		info = b.inferModelCapabilities(model.Name, model.Type)
	}

	// Create model metadata
	meta := &ai.ModelOptions{
		Label:    provider + "-" + model.Name,
		Supports: info.Supports,
		Versions: info.Versions,
	}

	// Create the model function based on model type
	switch model.Type {
	case "image":
		return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, func(
			ctx context.Context,
			input *ai.ModelRequest,
			cb func(context.Context, *ai.ModelResponseChunk) error,
		) (*ai.ModelResponse, error) {
			return b.generateImage(ctx, model.Name, input, cb)
		})
	default:
		return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, func(
			ctx context.Context,
			input *ai.ModelRequest,
			cb func(context.Context, *ai.ModelResponseChunk) error,
		) (*ai.ModelResponse, error) {
			return b.generateText(ctx, model.Name, input, cb)
		})
	}
}

// DefineEmbedder defines an embedder in the registry.
func (b *Bedrock) DefineEmbedder(g *genkit.Genkit, modelName string) ai.Embedder {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.initted {
		panic("bedrock: Init not called")
	}

	return genkit.DefineEmbedder(g, api.NewName(provider, modelName), nil, func(
		ctx context.Context,
		req *ai.EmbedRequest,
	) (*ai.EmbedResponse, error) {
		return b.embed(ctx, modelName, req)
	})
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Model returns the Model with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// inferModelCapabilities infers model capabilities based on model name and type.
// It strips any inference profile prefix before looking up capabilities.
func (b *Bedrock) inferModelCapabilities(modelName, modelType string) *ai.ModelInfo {
	// Strip inference profile prefix to get base model ID for capability lookup
	baseModelID := b.stripInferenceProfilePrefix(modelName)

	// Look up capabilities from the map
	caps, found := modelCapabilities[baseModelID]

	switch modelType {
	case "image":
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  false,
				Tools:      false,
				SystemRole: false,
				Media:      true, // Can output images
			},
		}
	case "embedding":
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  false,
				Tools:      false,
				SystemRole: false,
				Media:      false,
			},
		}
	default: // chat, text models
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      found && caps.Tools,
				SystemRole: true,
				Media:      found && caps.Multimodal,
			},
		}
	}
}

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

// generateImage handles image generation using Bedrock InvokeModel API
func (b *Bedrock) generateImage(ctx context.Context, modelName string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Extract prompt from the first message
	var prompt string
	if len(input.Messages) > 0 && len(input.Messages[0].Content) > 0 {
		if input.Messages[0].Content[0].IsText() {
			prompt = input.Messages[0].Content[0].Text
		}
	}

	if prompt == "" {
		return nil, fmt.Errorf("no text prompt found for image generation")
	}

	// Generate image based on model type
	switch {
	case strings.Contains(modelName, "titan-image"):
		return b.generateTitanImage(ctx, modelName, prompt, input.Config, cb)
	case strings.Contains(modelName, "stable-diffusion"), strings.Contains(modelName, "sd3-"), strings.Contains(modelName, "stable-image"):
		return b.generateStableDiffusionImage(ctx, modelName, prompt, input.Config, cb)
	case strings.Contains(modelName, "nova-canvas"):
		return b.generateNovaCanvasImage(ctx, modelName, prompt, input.Config, cb)
	default:
		return nil, fmt.Errorf("unsupported image generation model: %s", modelName)
	}
}

// generateTitanImage generates images using Amazon Titan Image Generator
func (b *Bedrock) generateTitanImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Prepare request body for Titan Image Generator
	requestBody := map[string]interface{}{
		"taskType": "TEXT_IMAGE",
		"textToImageParams": map[string]interface{}{
			"text": prompt,
		},
		"imageGenerationConfig": map[string]interface{}{
			"numberOfImages": 1,
			"height":         1024,
			"width":          1024,
			"cfgScale":       8.0,
			"seed":           0,
		},
	}

	// Apply config if provided
	if config != nil {
		if configMap, ok := config.(map[string]interface{}); ok {
			if imageConfig, exists := configMap["imageGenerationConfig"]; exists {
				if imgCfg, ok := imageConfig.(map[string]interface{}); ok {
					for k, v := range imgCfg {
						requestBody["imageGenerationConfig"].(map[string]interface{})[k] = v
					}
				}
			}
		}
	}

	// Marshal request
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call InvokeModel
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	response, err := b.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parse response
	var result struct {
		Images []string `json:"images"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(result.Images) == 0 {
		return nil, fmt.Errorf("no images generated")
	}

	// Create response with image data
	return &ai.ModelResponse{
		Message: &ai.Message{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,"+result.Images[0]),
			},
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// generateStableDiffusionImage generates images using Stability AI Stable Diffusion
func (b *Bedrock) generateStableDiffusionImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Prepare request body for Stable Diffusion
	requestBody := map[string]interface{}{
		"text_prompts": []map[string]interface{}{
			{
				"text":   prompt,
				"weight": 1.0,
			},
		},
		"cfg_scale":            7,
		"clip_guidance_preset": "FAST_BLUE",
		"height":               512,
		"width":                512,
		"samples":              1,
		"steps":                30,
	}

	// Apply config if provided
	if config != nil {
		if configMap, ok := config.(map[string]interface{}); ok {
			for k, v := range configMap {
				requestBody[k] = v
			}
		}
	}

	// Marshal request
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call InvokeModel
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	response, err := b.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parse response
	var result struct {
		Artifacts []struct {
			Base64       string `json:"base64"`
			FinishReason string `json:"finishReason"`
		} `json:"artifacts"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(result.Artifacts) == 0 {
		return nil, fmt.Errorf("no images generated")
	}

	// Create response with image data
	return &ai.ModelResponse{
		Message: &ai.Message{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,"+result.Artifacts[0].Base64),
			},
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// embed handles embedding generation using Bedrock InvokeModel API
func (b *Bedrock) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	var embeddings []*ai.Embedding

	// Process each document
	for _, doc := range req.Input {
		var inputText string

		// Extract text from document parts
		for _, part := range doc.Content {
			if part.IsText() {
				inputText += part.Text
			}
		}

		if inputText == "" {
			continue // Skip empty documents
		}

		// Prepare embedding request based on model
		var embedding []float32
		var err error

		switch {
		case strings.Contains(modelName, "titan"):
			embedding, err = b.getTitanEmbedding(ctx, modelName, inputText)
		case strings.Contains(modelName, "cohere"):
			embedding, err = b.getCohereEmbedding(ctx, modelName, inputText)
		default:
			return nil, fmt.Errorf("unsupported embedding model: %s", modelName)
		}

		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding: %w", err)
		}

		embeddings = append(embeddings, &ai.Embedding{
			Embedding: embedding,
		})
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}

// getTitanEmbedding generates embeddings using Amazon Titan embedding models
func (b *Bedrock) getTitanEmbedding(ctx context.Context, modelName, text string) ([]float32, error) {
	// Prepare request body for Titan embedding model
	requestBody := map[string]interface{}{
		"inputText": text,
	}

	// Marshal request
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call InvokeModel
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	response, err := b.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parse response
	var result struct {
		Embedding []float32 `json:"embedding"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return result.Embedding, nil
}

// getCohereEmbedding generates embeddings using Cohere embedding models
func (b *Bedrock) getCohereEmbedding(ctx context.Context, modelName, text string) ([]float32, error) {
	// Prepare request body for Cohere embedding model
	requestBody := map[string]interface{}{
		"texts":      []string{text},
		"input_type": "search_document",
	}

	// Marshal request
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call InvokeModel
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	response, err := b.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parse response
	var result struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return result.Embeddings[0], nil
}

// generateNovaCanvasImage generates images using Amazon Nova Canvas
func (b *Bedrock) generateNovaCanvasImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Prepare request body for Nova Canvas
	requestBody := map[string]interface{}{
		"taskType": "TEXT_IMAGE",
		"textToImageParams": map[string]interface{}{
			"text": prompt,
		},
		"imageGenerationConfig": map[string]interface{}{
			"numberOfImages": 1,
			"quality":        "standard",
			"height":         1024,
			"width":          1024,
			"cfgScale":       8.0,
			"seed":           0,
		},
	}

	// Apply config if provided
	if config != nil {
		if configMap, ok := config.(map[string]interface{}); ok {
			if imageConfig, exists := configMap["imageGenerationConfig"]; exists {
				if imgCfg, ok := imageConfig.(map[string]interface{}); ok {
					for k, v := range imgCfg {
						requestBody["imageGenerationConfig"].(map[string]interface{})[k] = v
					}
				}
			}
		}
	}

	// Marshal request
	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call InvokeModel
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	}

	response, err := b.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parse response (Nova Canvas uses similar format to Titan)
	var result struct {
		Images []string `json:"images"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(result.Images) == 0 {
		return nil, fmt.Errorf("no images generated")
	}

	// Create response with image data
	return &ai.ModelResponse{
		Message: &ai.Message{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,"+result.Images[0]),
			},
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// buildConverseInput converts Genkit ModelRequest to Bedrock ConverseInput
func (b *Bedrock) buildConverseInput(modelName string, input *ai.ModelRequest) (*bedrockruntime.ConverseInput, error) {
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
						var imageBlock *types.ContentBlockMemberImage

						// Parse data URL or direct content
						content := part.Text
						if strings.HasPrefix(content, "data:") {
							// Handle data URL format: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
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

						// Convert to appropriate image format based on MIME type
						var format types.ImageFormat
						switch mediaType {
						case "image/png":
							format = types.ImageFormatPng
						case "image/jpeg", "image/jpg":
							format = types.ImageFormatJpeg
						case "image/gif":
							format = types.ImageFormatGif
						case "image/webp":
							format = types.ImageFormatWebp
						default:
							// Default to PNG if unknown
							format = types.ImageFormatPng
						}

						// Decode base64 content
						imageData, err := base64.StdEncoding.DecodeString(content)
						if err != nil {
							// If decoding fails, try using the content directly
							imageData = []byte(content)
						}

						imageBlock = &types.ContentBlockMemberImage{
							Value: types.ImageBlock{
								Format: format,
								Source: &types.ImageSourceMemberBytes{
									Value: imageData,
								},
							},
						}

						contentBlocks = append(contentBlocks, imageBlock)
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

	// Set inference configuration
	if input.Config != nil {
		if configMap, ok := input.Config.(map[string]interface{}); ok {
			inferenceConfig := &types.InferenceConfiguration{}

			if maxTokens, ok := configMap["maxOutputTokens"].(int); ok {
				inferenceConfig.MaxTokens = aws.Int32(int32(maxTokens))
			} else if maxTokens, ok := configMap["max_tokens"].(int); ok {
				inferenceConfig.MaxTokens = aws.Int32(int32(maxTokens))
			}

			if temp, ok := configMap["temperature"].(float64); ok {
				inferenceConfig.Temperature = aws.Float32(float32(temp))
			}

			if topP, ok := configMap["topP"].(float64); ok {
				inferenceConfig.TopP = aws.Float32(float32(topP))
			}

			if stopSequences, ok := configMap["stopSequences"].([]string); ok {
				inferenceConfig.StopSequences = stopSequences
			}

			converseInput.InferenceConfig = inferenceConfig
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
	// Call Bedrock Converse API
	response, err := b.client.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse failed: %w", err)
	}

	// Convert response to Genkit format
	return b.convertResponse(response, originalInput), nil
}

// generateTextStream handles streaming text generation
func (b *Bedrock) generateTextStream(ctx context.Context, input *bedrockruntime.ConverseInput, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Convert ConverseInput to ConverseStreamInput
	streamInput := &bedrockruntime.ConverseStreamInput{
		ModelId:                      input.ModelId,
		Messages:                     input.Messages,
		System:                       input.System,
		InferenceConfig:              input.InferenceConfig,
		ToolConfig:                   input.ToolConfig,
		AdditionalModelRequestFields: input.AdditionalModelRequestFields,
	}

	// Call Bedrock ConverseStream API
	streamOutput, err := b.client.ConverseStream(ctx, streamInput)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse stream failed: %w", err)
	}
	defer func() {
		if closeErr := streamOutput.GetStream().Close(); closeErr != nil {
			// Log the error but don't fail the operation
			// In a real implementation, you might want to use a proper logger
			_ = closeErr
		}
	}()

	// Build final response
	var fullText strings.Builder
	var finalResponse *ai.ModelResponse
	var stopReason types.StopReason

	// Process stream events
	for event := range streamOutput.GetStream().Events() {
		switch e := event.(type) {

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			// Text delta received
			deltaEvent := e.Value
			if deltaEvent.Delta != nil {
				if textDelta, ok := deltaEvent.Delta.(*types.ContentBlockDeltaMemberText); ok {
					text := textDelta.Value
					fullText.WriteString(text)

					// Send chunk to callback
					chunk := &ai.ModelResponseChunk{
						Index: 0,
						Content: []*ai.Part{
							ai.NewTextPart(text),
						},
					}
					if err := cb(ctx, chunk); err != nil {
						return nil, fmt.Errorf("callback error: %w", err)
					}
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			// Message ended - prepare final response
			stopEvent := e.Value
			stopReason = stopEvent.StopReason

			finalResponse = &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewTextPart(fullText.String()),
					},
				},
				FinishReason: convertStopReasonToGenkit(stopReason),
			}

		}
	}

	// Return final response
	if finalResponse == nil {
		finalResponse = &ai.ModelResponse{
			Message: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewTextPart(fullText.String()),
				},
			},
			FinishReason: ai.FinishReasonStop,
		}
	}

	return finalResponse, nil
}

// convertResponse converts Bedrock response to Genkit format
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

// stripInferenceProfilePrefix removes the regional inference profile prefix
// from a model ID if present, returning the base model ID.
// Example: "us.anthropic.claude-3-haiku-20240307-v1:0" -> "anthropic.claude-3-haiku-20240307-v1:0"
func (b *Bedrock) stripInferenceProfilePrefix(modelID string) string {
	for _, prefix := range inferenceProfilePrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return strings.TrimPrefix(modelID, prefix)
		}
	}
	return modelID
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

// DefineCommonModels is a helper to define commonly used models
func DefineCommonModels(b *Bedrock, g *genkit.Genkit) map[string]ai.Model {
	models := make(map[string]ai.Model)

	// Text generation models
	claudeHaiku := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-3-haiku-20240307-v1:0",
		Type: "chat",
	}, nil)
	models["claude-haiku"] = claudeHaiku

	claudeSonnet := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-3-5-sonnet-20241022-v2:0",
		Type: "chat",
	}, nil)
	models["claude-sonnet"] = claudeSonnet

	// Claude 4 models
	claudeOpus4 := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-opus-4-20250514-v1:0",
		Type: "chat",
	}, nil)
	models["claude-opus-4"] = claudeOpus4

	claudeSonnet4 := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-sonnet-4-20250514-v1:0",
		Type: "chat",
	}, nil)
	models["claude-sonnet-4"] = claudeSonnet4

	// Claude 3.7 Sonnet
	claude37Sonnet := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Type: "chat",
	}, nil)
	models["claude-3-7-sonnet"] = claude37Sonnet

	// Amazon Nova models
	novaMicro := b.DefineModel(g, ModelDefinition{
		Name: "amazon.nova-micro-v1:0",
		Type: "chat",
	}, nil)
	models["nova-micro"] = novaMicro

	novaLite := b.DefineModel(g, ModelDefinition{
		Name: "amazon.nova-lite-v1:0",
		Type: "chat",
	}, nil)
	models["nova-lite"] = novaLite

	novaPro := b.DefineModel(g, ModelDefinition{
		Name: "amazon.nova-pro-v1:0",
		Type: "chat",
	}, nil)
	models["nova-pro"] = novaPro

	// Legacy models for backward compatibility
	titanText := b.DefineModel(g, ModelDefinition{
		Name: "amazon.titan-text-premier-v1:0",
		Type: "chat",
	}, nil)
	models["titan-text"] = titanText

	// Meta Llama models
	llama3_8b := b.DefineModel(g, ModelDefinition{
		Name: "meta.llama3-8b-instruct-v1:0",
		Type: "chat",
	}, nil)
	models["llama3-8b"] = llama3_8b

	llama3_1_8b := b.DefineModel(g, ModelDefinition{
		Name: "meta.llama3-1-8b-instruct-v1:0",
		Type: "chat",
	}, nil)
	models["llama3-1-8b"] = llama3_1_8b

	llama3_2_3b := b.DefineModel(g, ModelDefinition{
		Name: "meta.llama3-2-3b-instruct-v1:0",
		Type: "chat",
	}, nil)
	models["llama3-2-3b"] = llama3_2_3b

	// New Llama 4 models
	llama4Maverick := b.DefineModel(g, ModelDefinition{
		Name: "meta.llama4-maverick-17b-instruct-v1:0",
		Type: "chat",
	}, nil)
	models["llama4-maverick"] = llama4Maverick

	llama4Scout := b.DefineModel(g, ModelDefinition{
		Name: "meta.llama4-scout-17b-instruct-v1:0",
		Type: "chat",
	}, nil)
	models["llama4-scout"] = llama4Scout

	// DeepSeek R1 model
	deepseekR1 := b.DefineModel(g, ModelDefinition{
		Name: "deepseek.r1-v1:0",
		Type: "chat",
	}, nil)
	models["deepseek-r1"] = deepseekR1

	// Image generation models
	titanImage := b.DefineModel(g, ModelDefinition{
		Name: "amazon.titan-image-generator-v1",
		Type: "image",
	}, nil)
	models["titan-image"] = titanImage

	novaCanvas := b.DefineModel(g, ModelDefinition{
		Name: "amazon.nova-canvas-v1:0",
		Type: "image",
	}, nil)
	models["nova-canvas"] = novaCanvas

	return models
}

// DefineCommonEmbedders is a helper to define commonly used embedders
func DefineCommonEmbedders(b *Bedrock, g *genkit.Genkit) map[string]ai.Embedder {
	embedders := make(map[string]ai.Embedder)

	// Amazon Titan Embeddings
	titanEmbed := b.DefineEmbedder(g, "amazon.titan-embed-text-v1")
	embedders["titan-embed"] = titanEmbed

	titanEmbedV2 := b.DefineEmbedder(g, "amazon.titan-embed-text-v2:0")
	embedders["titan-embed-v2"] = titanEmbedV2

	titanMultimodal := b.DefineEmbedder(g, "amazon.titan-embed-image-v1")
	embedders["titan-multimodal"] = titanMultimodal

	// Cohere Embeddings
	cohereEmbed := b.DefineEmbedder(g, "cohere.embed-english-v3")
	embedders["cohere-embed"] = cohereEmbed

	cohereMultilingual := b.DefineEmbedder(g, "cohere.embed-multilingual-v3")
	embedders["cohere-multilingual"] = cohereMultilingual

	return embedders
}

// NewCachePointPart creates and returns a new ai.Part instance representing a cache point part
// with the default cache point type. A cache point should be inserted after a big static prompt
// that is reused across multiple requests to optimize token usage.
func NewCachePointPart() *ai.Part {
	return ai.NewCustomPart(map[string]any{
		bedrockCachePointTypeKey: types.CachePointTypeDefault,
	})
}

// CachePointType retrieves the CachePointType value from the Custom field of the given ai.Part.
// It returns the CachePointType and a boolean indicating whether the value was found and successfully asserted.
func CachePointType(part *ai.Part) (types.CachePointType, bool) {
	cachePointTypeVal, ok := part.Custom[bedrockCachePointTypeKey]
	if !ok {
		return "", false
	}
	cpt, ok := cachePointTypeVal.(types.CachePointType)
	return cpt, ok
}
