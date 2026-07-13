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

// Package main demonstrates document reranking with AWS Bedrock.
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

const defaultRerankModel = "cohere.rerank-v3-5:0"

func exampleRerankModelID() string {
	if modelID := os.Getenv("BEDROCK_RERANK_MODEL"); modelID != "" {
		return modelID
	}
	return defaultRerankModel
}

func main() {
	ctx := context.Background()

	bedrockPlugin := &bedrock.Bedrock{
		Region: os.Getenv("BEDROCK_REGION"),
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Starting reranking example...")

	modelID := exampleRerankModelID()
	log.Printf("Using model: %s", modelID)

	response, err := bedrock.Rerank(ctx, g, modelID, &ai.RerankerRequest{
		Query: ai.DocumentFromText("How do I configure authentication for AWS Bedrock?", nil),
		Documents: []*ai.Document{
			ai.DocumentFromText("Configure AWS credentials with environment variables, shared credentials files, IAM roles, or AWS SSO.", map[string]any{"id": "auth"}),
			ai.DocumentFromText("Amazon Titan Image Generator returns generated images as base64-encoded image data.", map[string]any{"id": "image"}),
			ai.DocumentFromText("Bedrock model access is managed in the AWS Bedrock console for each supported region.", map[string]any{"id": "model-access"}),
		},
		Options: &bedrock.RerankOptions{TopN: 2},
	})
	if err != nil {
		log.Fatalf("Error reranking documents: %v", err)
	}

	for i, doc := range response.Documents {
		score := 0.0
		if doc.Metadata != nil {
			score = doc.Metadata.Score
		}
		text := ""
		if len(doc.Content) > 0 && doc.Content[0] != nil {
			text = doc.Content[0].Text
		}
		log.Printf("Rank %d score=%.4f text=%s", i+1, score, text)
	}

	log.Println("Reranking example completed")
}
