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

// Package main demonstrates basic usage of the AWS Bedrock plugin
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

const defaultTextModel = "amazon.nova-lite-v1:0"

func exampleModelID() string {
	if modelID := os.Getenv("BEDROCK_MODEL"); modelID != "" {
		return modelID
	}
	return defaultTextModel
}

func main() {
	ctx := context.Background()

	// Initialize Bedrock plugin. Region can be omitted when the AWS SDK region
	// chain is configured through AWS_REGION, AWS_DEFAULT_REGION, or shared config.
	bedrockPlugin := &bedrock.Bedrock{
		Region: os.Getenv("BEDROCK_REGION"),
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Genkit initialized")

	log.Println("Starting basic Bedrock example...")

	// Define a chat model. Set BEDROCK_MODEL to use a regional/global
	// inference profile or another model your account can access.
	modelID := exampleModelID()
	chatModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: modelID,
		Type: "chat",
	}, nil)
	log.Printf("Using model: %s", modelID)

	// Example: Generate text (basic usage)
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(chatModel),
		ai.WithPrompt("What are the key benefits of using AWS Bedrock for AI applications?"),
		ai.WithConfig(&bedrock.Config{MaxTokens: 512}),
	)
	if err != nil {
		log.Printf("Error generating text: %v", err)
	} else {
		log.Printf("Generated response: %s", response.Text())
	}

	log.Println("Basic Bedrock example completed")
}
