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

// Live tests exercise generation, reasoning ("thinking"), streaming, and tool
// calls against real Bedrock endpoints. They are skipped by default and only
// run when the required model flags are passed, e.g.:
//
//	go test -run TestBedrockLive_ClaudeSync \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-model-claude=us.anthropic.claude-haiku-4-5-20251001-v1:0
//
//	go test -run TestBedrockLive_NovaSync \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-model-nova=amazon.nova-lite-v1:0
//
// They require AWS credentials in the environment and model access granted in
// the target region. Reasoning support is region- and model-scoped on Bedrock.
// These tests validate request/response shapes against live Bedrock behavior,
// not that any particular model is granted.

import (
	"context"
	"flag"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

var (
	testRegion      = flag.String("test-bedrock-region", "", "AWS region for Bedrock live tests (e.g. us-east-1)")
	testModelClaude = flag.String("test-bedrock-model-claude", "", "Thinking-capable Claude model ID (e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0)")
	testModelNova   = flag.String("test-bedrock-model-nova", "", "Amazon Nova text model ID for live tests (e.g. amazon.nova-lite-v1:0)")
)

// reasoningBudgetTokens is the extended-thinking budget. Bedrock requires it to
// be at least 1024, and MaxTokens must exceed it.
const reasoningBudgetTokens = 1024

// requireLiveClaude asserts the live-test prerequisites and skips otherwise. It
// returns a Genkit instance with the Bedrock plugin and a defined Claude model.
func requireLiveClaude(t *testing.T) (context.Context, *genkit.Genkit, ai.Model) {
	t.Helper()
	return requireLiveChatModel(t, *testModelClaude, "pass -test-bedrock-model-claude=<thinking-capable-model-id> to run")
}

// requireLiveNova asserts the live-test prerequisites and skips otherwise. It
// returns a Genkit instance with the Bedrock plugin and a defined Nova model.
func requireLiveNova(t *testing.T) (context.Context, *genkit.Genkit, ai.Model) {
	t.Helper()
	return requireLiveChatModel(t, *testModelNova, "pass -test-bedrock-model-nova=<nova-model-id> to run")
}

func requireLiveChatModel(t *testing.T, modelID, missingModelMessage string) (context.Context, *genkit.Genkit, ai.Model) {
	t.Helper()
	if *testRegion == "" {
		t.Skip("bedrock live tests skipped; pass -test-bedrock-region=<region>")
	}
	if modelID == "" {
		t.Skip(missingModelMessage)
	}
	ctx := context.Background()
	pb := &Bedrock{Region: *testRegion}
	g := genkit.Init(ctx, genkit.WithPlugins(pb))
	m := pb.DefineModel(g, ModelDefinition{
		Name: modelID,
		Type: "chat",
	}, nil)
	return ctx, g, m
}

// thinkingConfig enables Claude extended thinking via AdditionalModelRequestFields.
// Temperature is intentionally left unset — Bedrock rejects thinking requests
// that also set a custom temperature.
func thinkingConfig() *Config {
	return &Config{
		MaxTokens: reasoningBudgetTokens + 1024,
		AdditionalModelRequestFields: map[string]any{
			"thinking": map[string]any{
				"type":          "enabled",
				"budget_tokens": reasoningBudgetTokens,
			},
		},
	}
}

// firstReasoning returns the first reasoning part in a message, or nil.
func firstReasoning(msg *ai.Message) *ai.Part {
	if msg == nil {
		return nil
	}
	for _, p := range msg.Content {
		if p.IsReasoning() {
			return p
		}
	}
	return nil
}

// TestBedrockLive_ClaudeSync confirms a standard synchronous Claude Converse
// request works without enabling Bedrock reasoning.
func TestBedrockLive_ClaudeSync(t *testing.T) {
	ctx, g, m := requireLiveClaude(t)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("Reply with one short sentence about AWS Bedrock."),
		ai.WithConfig(&Config{MaxTokens: 64}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() == "" {
		t.Fatal("Claude sync response text is empty")
	}
}

// TestBedrockLive_NovaSync confirms a synchronous Nova Converse request works
// through the same generation path as the rest of the live matrix.
func TestBedrockLive_NovaSync(t *testing.T) {
	ctx, g, m := requireLiveNova(t)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("Reply with one short sentence about AWS Bedrock."),
		ai.WithConfig(&Config{MaxTokens: 64}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() == "" {
		t.Fatal("Nova sync response text is empty")
	}
}

// TestBedrockLive_ClaudeReasoningSync confirms a thinking-enabled request comes
// back with a signed reasoning part, and that the plain text answer is still
// surfaced via Text() (i.e. reasoning doesn't leak into normal output).
func TestBedrockLive_ClaudeReasoningSync(t *testing.T) {
	ctx, g, m := requireLiveClaude(t)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("What is 17 * 24? Think it through step by step, then give the answer."),
		ai.WithConfig(thinkingConfig()),
	)
	if err != nil {
		t.Fatal(err)
	}

	reasoning := firstReasoning(resp.Message)
	if reasoning == nil {
		t.Fatal("expected a reasoning part in the response; got none")
	}
	if sig := metadataBytes(reasoning.Metadata, reasoningSignatureMetadataKey); len(sig) == 0 {
		t.Error("reasoning part is missing its Bedrock signature")
	}
	if resp.Text() == "" {
		t.Error("final response text is empty")
	}
}

// TestBedrockLive_ClaudeReasoningRoundTrip is the real proof of the feature: it
// feeds a thinking response back as conversation history and confirms the
// follow-up turn is accepted. If the signed/redacted reasoning weren't
// round-tripped verbatim, Bedrock rejects the request.
func TestBedrockLive_ClaudeReasoningRoundTrip(t *testing.T) {
	ctx, g, m := requireLiveClaude(t)

	turn1 := ai.NewUserTextMessage("What is 17 * 24? Show your reasoning, then state the result.")
	resp1, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(turn1),
		ai.WithConfig(thinkingConfig()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if firstReasoning(resp1.Message) == nil {
		t.Fatal("first turn produced no reasoning part; cannot exercise round-trip")
	}

	// Replay the assistant turn (reasoning included) plus a follow-up question.
	resp2, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(
			turn1,
			resp1.Message,
			ai.NewUserTextMessage("Now multiply that result by 2."),
		),
		ai.WithConfig(thinkingConfig()),
	)
	if err != nil {
		t.Fatalf("follow-up turn rejected (reasoning round-trip likely broken): %v", err)
	}
	if resp2.Text() == "" {
		t.Error("follow-up response text is empty")
	}
}

// TestBedrockLive_ClaudeReasoningStream confirms reasoning deltas stream through
// to the callback and the final response carries an assembled reasoning part.
func TestBedrockLive_ClaudeReasoningStream(t *testing.T) {
	ctx, g, m := requireLiveClaude(t)

	var reasoningChunks, textChunks int
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("What is 17 * 24? Think it through, then answer."),
		ai.WithConfig(thinkingConfig()),
		ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
			for _, p := range c.Content {
				switch {
				case p.IsReasoning():
					reasoningChunks++
				case p.IsText():
					textChunks++
				}
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if reasoningChunks == 0 {
		t.Error("expected at least one reasoning chunk")
	}
	if firstReasoning(resp.Message) == nil {
		t.Error("final response is missing the assembled reasoning part")
	}
	if resp.Text() == "" {
		t.Error("final response text is empty")
	}
}

// TestBedrockLive_ClaudeStreamingToolCall confirms streamed tool-use blocks are
// reassembled into a complete tool request, delivered to the stream callback,
// and accepted by Genkit's tool execution loop.
func TestBedrockLive_ClaudeStreamingToolCall(t *testing.T) {
	ctx, g, m := requireLiveClaude(t)

	type weatherIn struct {
		Location string `json:"location" jsonschema:"description=City to look up"`
	}
	toolCalls := 0
	weatherTool := genkit.DefineTool(g, "get_streaming_weather",
		"Get the current weather for a city.",
		func(ctx *ai.ToolContext, input weatherIn) (string, error) {
			toolCalls++
			return "72F in San Francisco", nil
		})

	var textChunks, toolRequestChunks int
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithSystem("Use the get_streaming_weather tool whenever weather is requested."),
		ai.WithPrompt("Use the weather tool for San Francisco, then answer with the temperature."),
		ai.WithTools(weatherTool),
		ai.WithConfig(&Config{
			MaxTokens: 256,
		}),
		ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
			for _, p := range c.Content {
				switch {
				case p.IsToolRequest():
					toolRequestChunks++
					if p.ToolRequest.Name != "get_streaming_weather" {
						t.Errorf("streamed tool name = %q, want get_streaming_weather", p.ToolRequest.Name)
					}
					if p.ToolRequest.Ref == "" {
						t.Error("streamed tool request missing ref")
					}
					if p.ToolRequest.Input == nil {
						t.Error("streamed tool request missing input")
					}
				case p.IsText():
					textChunks++
				}
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if toolRequestChunks == 0 {
		t.Fatal("expected at least one streamed tool request chunk")
	}
	if toolCalls == 0 {
		t.Fatal("expected weather tool to be invoked")
	}
	if textChunks == 0 {
		t.Error("expected at least one streamed text chunk")
	}
	if !strings.Contains(resp.Text(), "72") {
		t.Errorf("final response = %q, want it to include 72", resp.Text())
	}
}
