// Copyright 2025 Xavier Portilla Edo
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

// Package main demonstrates tool calling capabilities with AWS Bedrock
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

const defaultToolModel = "amazon.nova-lite-v1:0"

func exampleModelID() string {
	if modelID := os.Getenv("BEDROCK_MODEL"); modelID != "" {
		return modelID
	}
	return defaultToolModel
}

// Example tool for getting current weather
func getCurrentWeather(location string, unit string) (string, error) {
	// Mock weather data - in a real implementation, you'd call a weather API
	switch strings.ToLower(location) {
	case "san francisco", "san francisco, ca":
		if unit == "fahrenheit" {
			return "Sunny, 72°F (22°C) with light breeze", nil
		}
		return "Sunny, 22°C (72°F) with light breeze", nil
	case "new york", "new york, ny":
		if unit == "fahrenheit" {
			return "Partly cloudy, 68°F (20°C) with high humidity", nil
		}
		return "Partly cloudy, 20°C (68°F) with high humidity", nil
	case "london", "london, uk":
		if unit == "fahrenheit" {
			return "Overcast, 59°F (15°C) with occasional drizzle", nil
		}
		return "Overcast, 15°C (59°F) with occasional drizzle", nil
	default:
		if unit == "fahrenheit" {
			return fmt.Sprintf("Weather data for %s: Partly cloudy, 70°F", location), nil
		}
		return fmt.Sprintf("Weather data for %s: Partly cloudy, 21°C", location), nil
	}
}

// Example tool for performing calculations
func calculate(operation string, a, b float64) (string, error) {
	var result float64
	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return "", errors.New("division by zero")
		}
		result = a / b
	default:
		return "", errors.New("unknown operation: " + operation)
	}
	return strconv.FormatFloat(result, 'f', -1, 64), nil
}

// Example tool for getting current time
func getCurrentTime(timezone string) (string, error) {
	now := time.Now()
	if timezone != "" {
		// In a real implementation, you'd handle timezone conversion properly
		// For this example, we'll just append the timezone name
		return fmt.Sprintf("%s (%s timezone)", now.Format("2006-01-02 15:04:05"), timezone), nil
	}
	return now.Format("2006-01-02 15:04:05 UTC"), nil
}

func main() {
	ctx := context.Background()
	temperature := float32(0.1)

	bedrockPlugin := &bedrock.Bedrock{
		Region: os.Getenv("BEDROCK_REGION"),
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	log.Println("Starting tool calling example...")

	// Define a chat model that supports tool calling. Set BEDROCK_MODEL to use
	// a regional/global inference profile or another model your account can access.
	modelID := exampleModelID()
	chatModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: modelID,
		Type: "chat",
	}, nil)
	log.Printf("Using model: %s", modelID)

	// Define weather tool
	weatherTool := genkit.DefineTool(g, "get_current_weather",
		"Get current weather information",
		func(ctx *ai.ToolContext, input struct {
			Location string `json:"location" jsonschema:"description=The city and state, e.g. San Francisco, CA"`
			Unit     string `json:"unit,omitempty" jsonschema:"description=The temperature unit (celsius or fahrenheit),enum=celsius,enum=fahrenheit"`
		}) (string, error) {
			unit := input.Unit
			if unit == "" {
				unit = "celsius"
			}
			return getCurrentWeather(input.Location, unit)
		})

	// Define calculation tool
	calcTool := genkit.DefineTool(g, "calculate",
		"Perform basic arithmetic operations",
		func(ctx *ai.ToolContext, input struct {
			Operation string  `json:"operation" jsonschema:"description=The arithmetic operation to perform,enum=add,enum=subtract,enum=multiply,enum=divide"`
			A         float64 `json:"a" jsonschema:"description=First number"`
			B         float64 `json:"b" jsonschema:"description=Second number"`
		}) (string, error) {
			return calculate(input.Operation, input.A, input.B)
		})

	// Define time tool
	timeTool := genkit.DefineTool(g, "get_current_time",
		"Get the current time in a specific timezone",
		func(ctx *ai.ToolContext, input struct {
			Timezone string `json:"timezone,omitempty" jsonschema:"description=The timezone (e.g. UTC, America/New_York)"`
		}) (string, error) {
			return getCurrentTime(input.Timezone)
		})

	// Test tool calling with multiple prompts
	prompts := []string{
		"What's the weather like in San Francisco?",
		"Calculate 25 * 4 + 10. First multiply 25 * 4, then add 10 to the result.",
		"What time is it right now?",
		"Get me the weather in New York and tell me what time it is.",
	}

	for i, prompt := range prompts {
		log.Printf("\n--- Test %d ---", i+1)
		log.Printf("Prompt: %s", prompt)

		response, err := genkit.Generate(ctx, g,
			ai.WithModel(chatModel),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart(prompt),
			)),
			ai.WithTools(weatherTool, calcTool, timeTool),
			ai.WithConfig(&bedrock.Config{
				Temperature: &temperature, // Lower temperature for more consistent tool usage.
				MaxTokens:   1000,
				ToolChoice:  bedrock.ToolChoiceAuto,
			}),
		)

		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}

		log.Printf("Response: %s", response.Text())

		// Log if the model used any tools
		if response.Message != nil && len(response.Message.Content) > 0 {
			hasToolUse := false
			for _, part := range response.Message.Content {
				if !part.IsToolResponse() {
					hasToolUse = true
					break
				}
			}
			if hasToolUse {
				log.Printf("✓ Model used tools in this response")
			} else {
				log.Printf("○ Model responded without using tools")
			}
		}
	}
	log.Println("Tool calling example completed")

	// Summary
	log.Printf("\n=== Summary ===")
	log.Printf("Tested %d different prompts with %s", len(prompts), modelID)
	log.Printf("Available tools: weather (%s), calculator (%s), time (%s)",
		"get_current_weather", "calculate", "get_current_time")
	log.Printf("The model should have automatically selected appropriate tools based on the prompts")
}
