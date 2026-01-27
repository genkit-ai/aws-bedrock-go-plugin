# AWS Bedrock Plugin for Genkit Go

A comprehensive AWS Bedrock plugin for Genkit Go that provides text generation, image generation, and embedding capabilities using AWS Bedrock foundation models via the Converse API.

## Features

- **Text Generation**: Support for multiple foundation models via AWS Bedrock Converse API
- **Image Generation**: Support for image generation models like Amazon Titan Image Generator
- **Embeddings**: Support for text embedding models from Amazon Titan and Cohere
- **Streaming**: Full streaming support for real-time responses
- **Tool Calling**: Complete function calling capabilities with schema validation and type conversion
- **Multimodal Support**: Support for text + image inputs (vision models)
- **Schema Management**: Automatic conversion between Genkit and AWS Bedrock schemas
- **Type Safety**: Robust type conversion for tool parameters (handles AWS document.Number types)

## Supported Models

### Text Generation Models (with Tool Calling Support)
- **Anthropic Claude 3/3.5/4/4.5**: Haiku, Sonnet, Opus (all versions)
- **Amazon Nova**: Micro, Lite, Pro
- **Meta Llama**: 3.1/3.2/3.3 (8B, 70B, 405B)
- **Mistral AI**: Large, 7B models
- **Amazon Titan**: Text Express, Premier

### Image Generation Models  
- **Amazon Titan Image Generator v1**
- **Amazon Titan Image Generator v2** (preview)

### Embedding Models
- **Amazon Titan Embeddings**: Text v1/v2, Multimodal v1
- **Cohere**: Embed English/Multilingual v3

### Multimodal Models (Text + Vision)
- All Claude 3/3.5/4 models
- Amazon Nova models

## Installation

```bash
go get github.com/xavidop/genkit-aws-bedrock-go
```

## Quick Start

## Initialize the Plugin
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
		Region: "us-east-1",
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
		genkit.WithDefaultModel("bedrock/anthropic.claude-3-haiku-20240307-v1:0"), // Set default model
        
        // Or with a global/regional inference profile
        // genkit.WithDefaultModel("bedrock/global.anthropic.claude-3-haiku-20240307-v1:0")
	)

    bedrock.DefineCommonModels(bedrockPlugin, g) // Optional: Define common models for easy access

	log.Println("Starting basic Bedrock example...")

	// Example: Generate text (basic usage)
	response, err := genkit.Generate(ctx, g,
		ai.WithPrompt("What are the key benefits of using AWS Bedrock for AI applications?"),
	)
	if err != nil {
		log.Printf("Error generating text: %v", err)
	} else {
		log.Printf("Generated response: %s", response.Text())
	}

	log.Println("Basic Bedrock example completed")
}

```

## Define Models and Generate Text
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

    // Initialize Bedrock plugin
    bedrockPlugin := &bedrock.Bedrock{
        Region: "us-east-1", // Optional, defaults to AWS_REGION or us-east-1
    }
    
    // Initialize Genkit
    g := genkit.Init(ctx,
        genkit.WithPlugins(bedrockPlugin),
    )
    
    // Define a Claude 3 model
    claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
        Name: "anthropic.claude-3-haiku-20240307-v1:0",
        Type: "text",
    }, nil)
    
    // Generate text
    response, err := genkit.Generate(ctx, g,
        ai.WithModel(claudeModel),
        ai.WithMessages(ai.NewUserMessage(
            ai.NewTextPart("Hello! How are you?"),
        )),
    )
    
    if err != nil {
        log.Fatal(err)
    }
    
    log.Println(response.Text())
}
```

## Configuration Options

The plugin supports various configuration options:

```go
bedrockPlugin := &bedrock.Bedrock{
    Region:         "us-west-2",           // AWS region
    MaxRetries:     3,                     // Max retry attempts
    RequestTimeout: 30 * time.Second,     // Request timeout
    AWSConfig:      customAWSConfig,      // Custom AWS config (optional)
}
```

### Available Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Region` | `string` | `"us-east-1"` | AWS region for Bedrock |
| `MaxRetries` | `int` | `3` | Maximum retry attempts |
| `RequestTimeout` | `time.Duration` | `30s` | Request timeout |
| `AWSConfig` | `*aws.Config` | `nil` | Custom AWS configuration |


## AWS Setup and Authentication

The plugin uses the standard AWS SDK v2 configuration methods:

### Authentication Methods
1. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"  
   export AWS_REGION="us-east-1"
   ```

2. **AWS Credentials File** (`~/.aws/credentials`):
   ```ini
   [default]
   aws_access_key_id = your-access-key
   aws_secret_access_key = your-secret-key
   region = us-east-1
   ```

3. **IAM Roles** (when running on AWS services like EC2, ECS, Lambda)

4. **AWS SSO/CLI** (`aws configure sso`)

### Required IAM Permissions

Create an IAM policy with these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/*"
            ]
        }
    ]
}
```

### Model Access

Some models require additional access requests:
1. Go to **AWS Bedrock Console** → **Model Access**
2. Request access for specific models (e.g., Claude, Llama)
3. Wait for approval (usually instant for most models)

## Examples Directory

The repository includes comprehensive examples:

- **`examples/basic/`** - Simple text generation
- **`examples/streaming/`** - Real-time streaming responses  
- **`examples/tool_calling/`** - Function calling with multiple tools
- **`examples/image_generation/`** - Image generation and file saving
- **`examples/embeddings/`** - Text embeddings and similarity
- **`examples/multimodal/`** - Vision models with image inputs
- **`examples/advanced_schemas/`** - Complex tool schemas and validation
- **`examples/prompt_caching`** - Several calls with prompt caching enabled to save costs

### Running Examples

```bash
# Clone the repository
git clone https://github.com/genkit-ai/aws-bedrock-go-plugin
cd genkit-aws-bedrock

# Run basic example
cd examples/basic
go run main.go

# Run tool calling example
cd ../tool_calling  
go run main.go

# Run image generation example
cd ../image_generation
go run main.go
```

## Features in Detail

### 🔧 Tool Calling
- **Schema Validation**: Automatic validation and type conversion
- **Type Safety**: Handles AWS `document.Number` to Go numeric types
- **Complex Schemas**: Support for nested objects, arrays, enums
- **Error Handling**: Robust error handling and fallbacks

### 🖼️ Image Support  
- **Input**: Supports base64 data URLs and binary data
- **Output**: Returns images as base64 data URLs
- **Formats**: PNG, JPEG, WebP, GIF support
- **Vision**: Text + image inputs for multimodal models

### 📡 Streaming
- **Real-time**: Token-by-token streaming responses
- **Efficient**: Low-latency streaming with proper buffering
- **Error Handling**: Stream error handling and recovery

### 🎯 Type Conversion
- **AWS Document Types**: Handles `document.Number`, `document.String`
- **Schema Mapping**: JSON Schema to AWS Bedrock schema conversion
- **Parameter Types**: String to number/boolean conversion for tools

## Performance and Best Practices


### Configuration Tips

```go
// For development/testing
config := map[string]interface{}{
    "temperature": 0.1,        // Low for consistent results
    "maxOutputTokens": 1000,   // Reasonable limit
}

// For creative tasks  
config := map[string]interface{}{
    "temperature": 0.8,        // Higher for creativity
    "topP": 0.9,              // Nucleus sampling
    "maxOutputTokens": 2000,   // Allow longer responses
}

// For tool calling
config := map[string]interface{}{
    "temperature": 0.1,        // Low for consistent tool usage
    "maxOutputTokens": 1000,   // Tools need space for responses
}
```

### Error Handling

```go
response, err := genkit.Generate(ctx, g, /* options */)
if err != nil {
    // Handle specific AWS errors
    if strings.Contains(err.Error(), "ValidationException") {
        log.Printf("Request validation error: %v", err)
    } else if strings.Contains(err.Error(), "ThrottlingException") {
        log.Printf("Rate limit exceeded: %v", err)
        // Implement retry logic
    } else {
        log.Printf("Generation error: %v", err)
    }
    return
}
```

### Prompt Caching

```go
// Prompt caching helps to save input token costs and reduce latency for repeated contexts.
// The first cache point must be defined after 1,024 tokens for most models.
// More about prompt caching: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
response, err := genkit.Generate(ctx, g,
    ai.WithMessages(
        ai.NewSystemMessage(
            ai.NewTextPart(sysprompt), // A big system prompt that is reused
            bedrock.NewCachePointPart(), // A cache point after the system prompt

        ),
        ai.NewUserTextMessage(input),
    ),
)
```

## Troubleshooting

### Common Issues

1. **"Access Denied" Errors**
   - Verify IAM permissions include `bedrock:InvokeModel`
   - Check AWS credentials are configured correctly
   - Ensure you're using the correct AWS region

2. **"Model Not Found" Errors**  
   - Request model access in AWS Bedrock Console
   - Verify model name spelling and version
   - Check model availability in your region

3. **"Validation Exception" Errors**
   - Check tool schema definitions
   - Verify parameter types match schema
   - Ensure required fields are provided

4. **"Throttling Exception" Errors**
   - Implement exponential backoff retry logic
   - Consider upgrading to higher rate limits
   - Distribute requests across time

## Semantic Versioning & Automated Releases

This project follows [Semantic Versioning](https://semver.org/) and uses automated releases based on [Conventional Commits](https://conventionalcommits.org/).

### How Releases Work

1. **Automatic Version Bumping**: Commit messages determine version increments
   - `feat:` → Minor version bump (new features)
   - `fix:` → Patch version bump (bug fixes)  
   - `BREAKING CHANGE:` → Major version bump (breaking changes)

2. **Automated Changelog**: Release notes are automatically generated from commit messages

3. **GitHub Releases**: Tagged releases with compiled artifacts and documentation

### For Contributors

When contributing, use conventional commit format:

```bash
# Feature (minor bump)
feat(models): add support for Claude 3.5 Sonnet

# Bug fix (patch bump)  
fix(streaming): resolve timeout issue with long responses

# Breaking change (major bump)
feat(api)!: change model configuration interface

BREAKING CHANGE: Model configuration now requires explicit region specification.
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow [Conventional Commits](https://conventionalcommits.org/) format
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request


## License

Apache 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Genkit team for the excellent Go framework
- AWS Bedrock team for the comprehensive AI model platform  
- The open source community for inspiration and feedback

---

**Built with ❤️ for the Genkit Go community**
