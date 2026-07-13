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

// Package main demonstrates streaming text generation with AWS Bedrock
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

const defaultStreamingModel = "amazon.nova-lite-v1:0"

func exampleModelID() string {
	if modelID := os.Getenv("BEDROCK_MODEL"); modelID != "" {
		return modelID
	}
	return defaultStreamingModel
}

func main() {
	ctx := context.Background()
	bedrockPlugin := &bedrock.Bedrock{
		Region: os.Getenv("BEDROCK_REGION"),
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Genkit initialized")

	log.Println("Starting streaming text generation example...")

	// Define a chat model. Set BEDROCK_MODEL to use a regional/global
	// inference profile or another model your account can access.
	modelID := exampleModelID()
	chatModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: modelID,
		Type: "chat",
	}, nil)
	log.Printf("Using model: %s", modelID)

	// Streaming callback to handle response chunks
	streamCallback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) > 0 {
			for _, part := range chunk.Content {
				if part.IsText() {
					fmt.Print(part.Text)
				}
			}
		}
		return nil
	}

	log.Println("Generating streaming response...")

	// Generate streaming response
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(chatModel),
		ai.WithPrompt("Write a short story about a robot learning to paint. Make it creative and engaging."),
		ai.WithConfig(&bedrock.Config{MaxTokens: 1000}),
		ai.WithStreaming(streamCallback),
	)

	if err != nil {
		log.Printf("Error in streaming generation: %v", err)
	} else {
		fmt.Println() // New line after streaming
		log.Printf("Streaming generation completed. Final response length: %d characters", len(response.Text()))
	}

	log.Println("Streaming example completed")
}
