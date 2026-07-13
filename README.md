# AWS Bedrock Plugin for Genkit Go

Genkit Go plugin for Amazon Bedrock. It supports Converse text generation,
streaming, tool calling, multimodal and document inputs, image generation,
embeddings, prompt caching, and Cohere reranking through Bedrock.

## Features

- **Text generation** through the Bedrock Converse API.
- **Streaming** with text, tool-use, and reasoning block reassembly.
- **Tool calling** with schema conversion and typed tool input coercion.
- **Multimodal inputs** for supported vision and document models.
- **Image generation** for Titan Image, Nova Canvas, Stable Diffusion XL, and modern Stability models.
- **Embeddings** for Titan text, Titan multimodal, Cohere text/image, and Nova text models.
- **Reranking** with Cohere Rerank models through Bedrock `InvokeModel`.
- **Prompt caching** with Bedrock cache point parts.
- **Inference profiles** for regional and global Bedrock profile IDs.

## Installation

```bash
go get github.com/xavidop/genkit-aws-bedrock-go
```

## Quick Start

```go
package main

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

func main() {
	ctx := context.Background()

	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1", // Optional if the AWS SDK region chain is configured.
	}
	g := genkit.Init(ctx, genkit.WithPlugins(bedrockPlugin))

	model := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "amazon.nova-lite-v1:0",
		Type: "chat",
	}, nil)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("What are the key benefits of using AWS Bedrock?"),
		ai.WithConfig(&bedrock.Config{MaxTokens: 512}),
	)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(resp.Text())
}
```

You can also register common aliases:

```go
models := bedrock.DefineCommonModels(bedrockPlugin, g)
embedders := bedrock.DefineCommonEmbedders(bedrockPlugin, g)

resp, err := genkit.Generate(ctx, g,
	ai.WithModel(models["nova-lite"]),
	ai.WithPrompt("Summarize what Bedrock does."),
)
_ = embedders
_ = resp
_ = err
```

## AWS Configuration

The plugin uses the AWS SDK for Go v2 configuration chain. Set `Bedrock.Region`
for an explicit region, or leave it empty and configure one with `AWS_REGION`,
`AWS_DEFAULT_REGION`, shared config, SSO, or the runtime environment. If no
region is resolved, initialization fails with a clear error.

```go
bedrockPlugin := &bedrock.Bedrock{
	Region:         "us-west-2",
	MaxRetries:     3,
	RequestTimeout: 30 * time.Second,
	AWSConfig:      customAWSConfig,
}
```

| Option | Default | Description |
| --- | --- | --- |
| `Region` | AWS SDK region chain | Optional explicit region override. |
| `MaxRetries` | `3` | AWS SDK retry attempts when loading default config. |
| `RequestTimeout` | `30s` | Per-call timeout for generation, embedding, image, and rerank calls. |
| `AWSConfig` | `nil` | Full AWS SDK config override for credentials, endpoint, HTTP client, or tests. |

Required permissions usually include:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "*"
}
```

Model access is managed per AWS account and region in the Bedrock console.
Request access for the model families you plan to use before running examples
or live tests.

## Models and Inference Profiles

Pass direct Bedrock model IDs or inference profile IDs to `DefineModel`,
`DefineEmbedder`, `genkit.WithDefaultModel`, and `Rerank`.

```go
// Direct model ID.
"amazon.nova-lite-v1:0"

// Regional or global inference profile ID.
"us.anthropic.claude-3-7-sonnet-20250219-v1:0"
"global.anthropic.claude-haiku-4-5-20251001-v1:0"
```

The plugin preserves the full model ID when calling Bedrock. For local Genkit
metadata only, it strips known profile prefixes (`us.`, `eu.`, `apac.`, `jp.`,
`au.`, `global.`, `us-gov.`) before looking up capability metadata. Unknown
chat models remain callable and are marked unstable in metadata.

## Generation Configuration

Use `bedrock.Config` for typed Converse configuration:

```go
temperature := float32(0.2)
topP := float32(0.9)

resp, err := genkit.Generate(ctx, g,
	ai.WithModel(model),
	ai.WithPrompt("Answer briefly."),
	ai.WithConfig(&bedrock.Config{
		MaxTokens:    512,
		Temperature:  &temperature,
		TopP:         &topP,
		ToolChoice:   bedrock.ToolChoiceAuto,
		StopSequences: []string{"END"},
	}),
)
```

`ai.GenerationCommonConfig` and legacy `map[string]any` configs are still
accepted for compatibility. Use `AdditionalModelRequestFields` for
model-specific Converse fields such as Claude extended thinking:

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModel(model),
	ai.WithPrompt("Think carefully, then answer."),
	ai.WithConfig(&bedrock.Config{
		MaxTokens: 4096,
		AdditionalModelRequestFields: map[string]any{
			"thinking": map[string]any{
				"type":          "enabled",
				"budget_tokens": 1024,
			},
		},
	}),
)
```

## Tool Calling

Define tools with Genkit and pass them to `genkit.Generate`. `ToolChoice` may be
`auto`, `any`, `required`, `none`, or the name of a declared tool.

```go
weatherTool := genkit.DefineTool(g, "get_weather", "Get weather for a city.",
	func(ctx *ai.ToolContext, input struct {
		Location string `json:"location"`
	}) (string, error) {
		return "Sunny, 22C", nil
	})

resp, err := genkit.Generate(ctx, g,
	ai.WithModel(model),
	ai.WithPrompt("Use the weather tool for London."),
	ai.WithTools(weatherTool),
	ai.WithConfig(&bedrock.Config{
		MaxTokens:  512,
		ToolChoice: "get_weather",
	}),
)
```

## Image Generation

Define image models with `Type: "image"`. Generated images are returned as
`image/png` media parts with `data:image/png;base64,...` URLs.

Supported families:

- `amazon.titan-image-generator-*`
- `amazon.nova-canvas-*`
- `stability.stable-diffusion-xl-*`
- `stability.sd3-*`
- `stability.stable-image-*`

Titan Image and Nova Canvas accept nested `imageGenerationConfig` overrides.
Stable Diffusion XL accepts flat Stability fields. Modern Stability models
currently use fixed plugin defaults and ignore extra config fields.

## Embeddings

Define embedders with the Bedrock model ID:

```go
titan := bedrockPlugin.DefineEmbedder(g, "amazon.titan-embed-text-v2:0")

resp, err := genkit.Embed(ctx, g,
	ai.WithEmbedder(titan),
	ai.WithTextDocs("Bedrock provides managed foundation models."),
)
```

Supported families:

- Titan text: `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`
- Titan multimodal: `amazon.titan-embed-image-v1`
- Cohere text and image: `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`
- Nova text: `amazon.nova-embed-text-v1:0`

## Reranking

Genkit Go does not yet expose a first-class reranker action, so this plugin
provides `Rerank`. It reuses the initialized Bedrock plugin client and returns
documents ordered by score with `ai.RankedDocumentMetadata`.

```go
resp, err := bedrock.Rerank(ctx, g, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
	Query: ai.DocumentFromText("Which document explains Bedrock authentication?", nil),
	Documents: []*ai.Document{
		ai.DocumentFromText("Configure AWS credentials with environment variables or AWS SSO.", nil),
		ai.DocumentFromText("Titan Image Generator returns base64-encoded PNG data.", nil),
	},
	Options: &bedrock.RerankOptions{TopN: 1},
})
if err != nil {
	log.Fatal(err)
}
for _, doc := range resp.Documents {
	log.Printf("score=%.3f text=%s", doc.Metadata.Score, doc.Content[0].Text)
}
```

## Prompt Caching

Use `bedrock.NewCachePointPart()` in system or message content where Bedrock
supports prompt caching. Cache points are passed through to Converse as
Bedrock cache point blocks.

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModel(model),
	ai.WithMessages(
		ai.NewSystemMessage(
			ai.NewTextPart(largeReusablePrompt),
			bedrock.NewCachePointPart(),
		),
		ai.NewUserTextMessage("Summarize the policy."),
	),
)
if err == nil {
	log.Println("cache read tokens:", resp.Usage.CachedContentTokens)
}
```

## Media and Document Inputs

Media inputs must use a supported MIME type and a base64 data URL or bare
base64 payload. Remote URLs, raw non-base64 content, missing content types, and
unknown MIME types are rejected before calling Bedrock.

Supported document MIME types include PDF, CSV, DOC, DOCX, XLS, XLSX, HTML,
plain text, and Markdown. Supported image inputs include common Bedrock image
formats such as PNG, JPEG, WebP, and GIF, depending on the target model.

## Examples

```bash
git clone https://github.com/genkit-ai/aws-bedrock-go-plugin
cd aws-bedrock-go-plugin

go run ./examples/basic
go run ./examples/streaming
go run ./examples/tool_calling
go run ./examples/image_generation
go run ./examples/embeddings
go run ./examples/reranking
(cd examples/multimodal && go run .)
go run ./examples/document /path/to/document.pdf
go run ./examples/prompt_caching
```

The examples make live Bedrock calls. Configure AWS credentials, region, and
model access before running them. Set `BEDROCK_REGION` to override the AWS SDK
region chain for any example.

Model overrides:

- `BEDROCK_MODEL`: text, streaming, tool, advanced schema, document, and multimodal fallback.
- `BEDROCK_VISION_MODEL`: multimodal image-input model.
- `BEDROCK_DOCUMENT_MODEL`: document/PDF model.
- `BEDROCK_PROMPT_CACHING_MODEL`: prompt-caching-capable model. Required for `examples/prompt_caching` because support and marketplace access vary by model/account.
- `BEDROCK_EMBED_MODEL`: embedding model.
- `BEDROCK_IMAGE_MODEL`: image generation model. Required for `examples/image_generation` because image model IDs vary by region and model lifecycle.
- `BEDROCK_RERANK_MODEL`: reranking model.

## Opt-in Live Tests

Live tests are skipped by default. Pass `-test-bedrock-region` and the model
flag for each family you want to exercise.

```bash
go test . -run TestBedrockLive_ClaudeSync \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-model-claude=us.anthropic.claude-haiku-4-5-20251001-v1:0

go test . -run TestBedrockLive_NovaSync \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-model-nova=amazon.nova-lite-v1:0

go test . -run TestBedrockLive_ClaudeStreamingToolCall \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-model-claude=us.anthropic.claude-haiku-4-5-20251001-v1:0

go test . -run 'TestBedrockLive_.*Image' \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-image-titan=amazon.titan-image-generator-v2:0 \
  -test-bedrock-image-nova-canvas=amazon.nova-canvas-v1:0 \
  -test-bedrock-image-sdxl=stability.stable-diffusion-xl-v1:0 \
  -test-bedrock-image-modern-stability=stability.sd3-large-v1:0

go test . -run 'TestBedrockLive_.*Embed' \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-embed-titan-text=amazon.titan-embed-text-v1 \
  -test-bedrock-embed-titan-text-v2=amazon.titan-embed-text-v2:0 \
  -test-bedrock-embed-titan-multimodal=amazon.titan-embed-image-v1 \
  -test-bedrock-embed-cohere-text=cohere.embed-english-v3 \
  -test-bedrock-embed-cohere-multilingual=cohere.embed-multilingual-v3 \
  -test-bedrock-embed-nova=amazon.nova-embed-text-v1:0

go test . -run TestBedrockLive_CohereRerank \
  -test-bedrock-region=us-east-1 \
  -test-bedrock-rerank-model=cohere.rerank-v3-5:0
```

## Known Limitations

- Reranking is exposed as `bedrock.Rerank` because Genkit Go does not currently expose a first-class reranker primitive.
- Live model availability varies by AWS region, account, and model-access grants.
- Image generation config shapes differ by model family; use the family-specific Bedrock fields accepted by the target model.
- Nova embeddings are currently text-only in this plugin.
- Bedrock reasoning metadata is provider-specific and should be replayed only through the Bedrock conversation that produced it.

## Troubleshooting

- **No region resolved**: set `Bedrock.Region`, `AWS_REGION`, `AWS_DEFAULT_REGION`, or a region in `~/.aws/config`.
- **Access denied**: check AWS credentials, IAM permissions, and Bedrock model access.
- **Model not found or invalid model identifier**: verify the model ID, inference profile ID, account access, and region availability.
- **ValidationException**: check media MIME types, tool schemas, config shape, and model-specific Bedrock requirements.
- **ThrottlingException**: reduce concurrency, retry with backoff, or request higher Bedrock quotas.

## Contributing

Use Conventional Commits for changes:

```bash
feat(models): add support for a new Bedrock model
fix(streaming): handle tool-use stream deltas
docs(readme): refresh live test commands
```

## License

Apache 2.0. See [LICENSE](LICENSE).
