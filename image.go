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

	prompt := imagePrompt(input)
	if prompt == "" {
		return nil, fmt.Errorf("no text prompt found for image generation")
	}

	// Generate image based on model type
	var images []string
	var err error
	switch {
	case strings.Contains(modelName, "titan-image"):
		images, err = b.generateTitanImage(ctx, modelName, prompt, input.Config, cb)
	case strings.Contains(modelName, "nova-canvas"):
		images, err = b.generateNovaCanvasImage(ctx, modelName, prompt, input.Config, cb)
	case isModernStabilityImageModel(modelName):
		images, err = b.generateModernStabilityImage(ctx, modelName, prompt, input.Config, cb)
	case strings.Contains(modelName, "stable-diffusion"):
		images, err = b.generateStableDiffusionImage(ctx, modelName, prompt, input.Config, cb)
	default:
		return nil, fmt.Errorf("unsupported image generation model: %s", modelName)
	}
	if err != nil {
		return nil, err
	}
	return imageResponse(input, images)
}

func imagePrompt(input *ai.ModelRequest) string {
	if input == nil {
		return ""
	}
	for i := len(input.Messages) - 1; i >= 0; i-- {
		msg := input.Messages[i]
		if msg == nil || msg.Role != ai.RoleUser {
			continue
		}
		var prompt strings.Builder
		for _, part := range msg.Content {
			if part != nil && part.IsText() {
				prompt.WriteString(part.Text)
			}
		}
		if prompt.Len() > 0 {
			return prompt.String()
		}
	}
	return ""
}

func isModernStabilityImageModel(modelName string) bool {
	return strings.Contains(modelName, "sd3-") || strings.Contains(modelName, "stable-image")
}

func imageResponse(input *ai.ModelRequest, images []string) (*ai.ModelResponse, error) {
	if len(images) == 0 {
		return nil, fmt.Errorf("no images generated")
	}
	parts := make([]*ai.Part, 0, len(images))
	for _, image := range images {
		if image == "" {
			continue
		}
		parts = append(parts, ai.NewMediaPart("image/png", "data:image/png;base64,"+image))
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("no images generated")
	}
	return &ai.ModelResponse{
		Request: input,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: parts,
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// generateTitanImage generates images using Amazon Titan Image Generator
func (b *Bedrock) generateTitanImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) ([]string, error) {
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
	return result.Images, nil
}

// generateStableDiffusionImage generates images using Stability AI Stable Diffusion
func (b *Bedrock) generateStableDiffusionImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) ([]string, error) {
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
	images := make([]string, 0, len(result.Artifacts))
	for _, artifact := range result.Artifacts {
		if artifact.Base64 != "" {
			images = append(images, artifact.Base64)
		}
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("no images generated")
	}
	return images, nil
}

func (b *Bedrock) generateModernStabilityImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) ([]string, error) {
	requestBody := map[string]interface{}{
		"prompt":        prompt,
		"output_format": "png",
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

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

	var result struct {
		Images        []string  `json:"images"`
		FinishReasons []*string `json:"finish_reasons"`
	}

	if err := json.Unmarshal(response.Body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	for _, reason := range result.FinishReasons {
		if reason != nil && *reason != "" && *reason != "SUCCESS" {
			return nil, fmt.Errorf("image generation finished with reason: %s", *reason)
		}
	}
	if len(result.Images) == 0 {
		return nil, fmt.Errorf("no images generated")
	}
	return result.Images, nil
}

func (b *Bedrock) generateNovaCanvasImage(ctx context.Context, modelName, prompt string, config any, cb func(context.Context, *ai.ModelResponseChunk) error) ([]string, error) {
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
	return result.Images, nil
}
