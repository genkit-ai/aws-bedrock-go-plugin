package main

import (
	"context"
	_ "embed"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

func examplePromptCachingModelID() string {
	if modelID := os.Getenv("BEDROCK_PROMPT_CACHING_MODEL"); modelID != "" {
		return modelID
	}
	return ""
}

// A prompt has to be big enough for being cached
// https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models
//
//go:embed sysprompt.md
var sysprompt string

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

	// Define a prompt-caching-capable model. Set BEDROCK_PROMPT_CACHING_MODEL
	// to a regional/global inference profile or another model your account can
	// access.
	modelID := examplePromptCachingModelID()
	if modelID == "" {
		log.Fatal("Set BEDROCK_PROMPT_CACHING_MODEL to a prompt-caching-capable model your account can access, for example a supported Claude inference profile")
	}
	chatModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: modelID,
		Type: "chat",
	}, nil)
	log.Printf("Using model: %s", modelID)

	// User prompts
	inputs := []string{
		"Write a four-line poem about lost love",
		"Write a four-line poem about autumn",
	}

	// call several times in a row to verify caching works
	for _, input := range inputs {
		response, err := genkit.Generate(ctx, g,
			ai.WithModel(chatModel),
			ai.WithMessages(
				ai.NewSystemMessage(
					ai.NewTextPart(sysprompt),
					bedrock.NewCachePointPart(), // add a cache point after the system prompt
				),
				ai.NewUserTextMessage(input),
			),
			ai.WithConfig(&bedrock.Config{MaxTokens: 512}),
		)
		if err != nil {
			log.Fatal(err)
		}
		log.Println("Request:", input)
		log.Printf("Response:\n%s\n", response.Text())
		log.Println("Tokens read from cache:", response.Usage.CachedContentTokens)
	}
}
