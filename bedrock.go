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
	"fmt"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// Bedrock provides configuration options for the AWS Bedrock plugin.
type Bedrock struct {
	Region         string        // AWS region override (optional; otherwise resolved by the AWS SDK)
	MaxRetries     int           // Maximum number of retries (default: 3)
	RequestTimeout time.Duration // Request timeout (default: 30s)
	AWSConfig      *aws.Config   // Custom AWS config (optional)

	mu      sync.Mutex // Mutex to control access
	client  BedrockClient
	initted bool // Whether the plugin has been initialized
}

// Name returns the provider name.
func (b *Bedrock) Name() string {
	return provider
}

// Init initializes the AWS Bedrock plugin.
// This method follows the same pattern as the Ollama plugin.
func (b *Bedrock) Init(ctx context.Context) []api.Action {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.initted {
		panic("bedrock: Init already called")
	}

	// Set defaults
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
		loadOptions := []func(*config.LoadOptions) error{
			config.WithRetryMaxAttempts(b.MaxRetries),
		}
		if b.Region != "" {
			loadOptions = append(loadOptions, config.WithRegion(b.Region))
		}

		// Load default AWS configuration
		awsConfig, err = config.LoadDefaultConfig(ctx, loadOptions...)
		if err != nil {
			panic(fmt.Sprintf("bedrock: failed to load AWS config: %v", err))
		}
	}

	if awsConfig.Region == "" {
		panic("bedrock: no AWS region resolved; set Bedrock.Region, AWS_REGION, AWS_DEFAULT_REGION, or a region in ~/.aws/config")
	}

	// Create Bedrock Runtime client
	b.client = bedrockruntime.NewFromConfig(awsConfig)

	b.initted = true

	return []api.Action{}
}

func (b *Bedrock) withRequestTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if b == nil {
		return ctx, func() {}
	}
	return withRequestTimeout(ctx, b.RequestTimeout)
}

func withRequestTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, timeout)
}

// DefineModel defines a model in the registry.
// This follows the same pattern as the Anthropic plugin's DefineModel method.
func (b *Bedrock) DefineModel(g *genkit.Genkit, model ModelDefinition, info *ai.ModelInfo) ai.Model {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.initted {
		panic("bedrock: Init not called")
	}

	providedInfo := info != nil

	// Auto-detect model capabilities if not provided
	if info == nil {
		info = b.inferModelCapabilities(model.Name, model.Type)
	} else {
		inferred := b.inferModelCapabilities(model.Name, model.Type)
		copyInfo := *info
		if copyInfo.Supports == nil {
			copyInfo.Supports = inferred.Supports
		}
		if copyInfo.Stage == "" {
			copyInfo.Stage = inferred.Stage
		}
		info = &copyInfo
	}
	label := provider + "-" + model.Name
	if providedInfo && info.Label != "" {
		label = info.Label
	}
	configSchemaMap := info.ConfigSchema
	if configSchemaMap == nil {
		configSchemaMap = configSchema()
	}

	// Create model metadata
	meta := &ai.ModelOptions{
		Label:        label,
		Supports:     info.Supports,
		Stage:        info.Stage,
		Versions:     info.Versions,
		ConfigSchema: configSchemaMap,
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

	// Amazon Nova Embeddings
	novaEmbed := b.DefineEmbedder(g, "amazon.nova-embed-text-v1:0")
	embedders["nova-embed"] = novaEmbed

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
