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

func (b *Bedrock) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("embed request is nil")
	}

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
