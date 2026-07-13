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

// Package main demonstrates embedding generation with AWS Bedrock
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

const defaultEmbedModel = "amazon.titan-embed-text-v2:0"

func exampleEmbedModelID() string {
	if modelID := os.Getenv("BEDROCK_EMBED_MODEL"); modelID != "" {
		return modelID
	}
	return defaultEmbedModel
}

func main() {
	ctx := context.Background()

	// Initialize Bedrock plugin
	bedrockPlugin := &bedrock.Bedrock{
		Region: os.Getenv("BEDROCK_REGION"),
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Starting embedding generation example...")

	// Define an embedding model. Other supported IDs include:
	// - amazon.titan-embed-text-v2:0
	// - amazon.titan-embed-image-v1
	// - cohere.embed-english-v3
	// - cohere.embed-multilingual-v3
	// - amazon.nova-embed-text-v1:0
	modelID := exampleEmbedModelID()
	embedder := bedrockPlugin.DefineEmbedder(g, modelID)
	log.Printf("Using embedder: %s", modelID)

	// Example texts for embedding
	texts := []string{
		"Machine learning is a subset of artificial intelligence.",
		"Natural language processing helps computers understand human language.",
		"Computer vision enables machines to interpret visual information.",
		"Deep learning uses neural networks with multiple layers.",
	}

	for i, text := range texts {
		log.Printf("Generating embedding for text %d: %s", i+1, text)

		// Generate embedding
		response, err := genkit.Embed(ctx, g,
			ai.WithEmbedder(embedder),
			ai.WithTextDocs(text),
		)

		if err != nil {
			log.Printf("Error generating embedding for text %d: %v", i+1, err)
			continue
		}

		// Process embedding vector
		if response != nil && len(response.Embeddings) > 0 {
			embedding := response.Embeddings[0]
			log.Printf("Generated embedding with %d dimensions", len(embedding.Embedding))

			// Example: Print first few dimensions
			if len(embedding.Embedding) > 5 {
				log.Printf("First 5 dimensions: %.4f, %.4f, %.4f, %.4f, %.4f",
					embedding.Embedding[0], embedding.Embedding[1], embedding.Embedding[2],
					embedding.Embedding[3], embedding.Embedding[4])
			}
		}

		log.Printf("Completed embedding for text %d", i+1)
	}

	log.Println("Embedding generation example completed")
}
