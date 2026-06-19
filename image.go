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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
)

// generateImage handles image generation using Bedrock InvokeModel API
func (b *Bedrock) generateImage(ctx context.Context, modelName string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	if input == nil {
		return nil, fmt.Errorf("model request is nil")
	}

	// Extract prompt from the first message
	var prompt string
	if len(input.Messages) > 0 && len(input.Messages[0].Content) > 0 {
		part := input.Messages[0].Content[0]
		if part != nil && part.IsText() {
			prompt = part.Text
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

	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

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

	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

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

	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

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
