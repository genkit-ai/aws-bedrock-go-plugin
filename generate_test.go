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

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
)

// ---- convertStopReasonToGenkit ----------------------------------------------

func TestConvertStopReasonToGenkit(t *testing.T) {
	tests := []struct {
		reason types.StopReason
		want   ai.FinishReason
	}{
		{types.StopReasonEndTurn, ai.FinishReasonStop},
		{types.StopReasonMaxTokens, ai.FinishReasonLength},
		{types.StopReasonStopSequence, ai.FinishReasonStop},
		{types.StopReasonToolUse, ai.FinishReasonStop},
		{types.StopReasonContentFiltered, ai.FinishReasonBlocked},
		{types.StopReasonGuardrailIntervened, ai.FinishReasonBlocked},
		{types.StopReasonModelContextWindowExceeded, ai.FinishReasonLength},
		{types.StopReasonMalformedModelOutput, ai.FinishReasonOther},
		{types.StopReasonMalformedToolUse, ai.FinishReasonOther},
		{"", ai.FinishReasonOther},
	}
	for _, tt := range tests {
		got := convertStopReasonToGenkit(tt.reason)
		if got != tt.want {
			t.Errorf("convertStopReasonToGenkit(%q) = %q, want %q", tt.reason, got, tt.want)
		}
	}
}

// ---- buildInferenceConfig ---------------------------------------------------

func TestBuildInferenceConfig_NilAndZero(t *testing.T) {
	if buildInferenceConfig(nil) != nil {
		t.Error("nil Config should return nil InferenceConfiguration")
	}
	if buildInferenceConfig(&Config{}) != nil {
		t.Error("zero Config should return nil InferenceConfiguration")
	}
}

func TestBuildInferenceConfig_Fields(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)

	cfg := &Config{
		MaxTokens:     512,
		Temperature:   &temp,
		TopP:          &topP,
		StopSequences: []string{"STOP", "END"},
	}
	ic := buildInferenceConfig(cfg)
	if ic == nil {
		t.Fatal("expected non-nil InferenceConfiguration")
	}
	if ic.MaxTokens == nil || *ic.MaxTokens != 512 {
		t.Errorf("MaxTokens = %v, want 512", ic.MaxTokens)
	}
	if ic.Temperature == nil || *ic.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", ic.Temperature)
	}
	if ic.TopP == nil || *ic.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", ic.TopP)
	}
	if len(ic.StopSequences) != 2 || ic.StopSequences[0] != "STOP" {
		t.Errorf("StopSequences = %v, want [STOP END]", ic.StopSequences)
	}
}

func TestBuildInferenceConfig_PartialFields(t *testing.T) {
	// Only MaxTokens — should produce non-nil with just that field.
	ic := buildInferenceConfig(&Config{MaxTokens: 256})
	if ic == nil || ic.MaxTokens == nil || *ic.MaxTokens != 256 {
		t.Fatalf("MaxTokens-only config: ic = %v", ic)
	}
	if ic.Temperature != nil || ic.TopP != nil || len(ic.StopSequences) != 0 {
		t.Error("unexpected non-nil optional fields")
	}

	// Only StopSequences.
	ic = buildInferenceConfig(&Config{StopSequences: []string{"stop"}})
	if ic == nil || len(ic.StopSequences) != 1 {
		t.Fatalf("StopSequences-only config: ic = %v", ic)
	}
	if ic.MaxTokens != nil {
		t.Fatalf("StopSequences-only MaxTokens = %v, want nil", *ic.MaxTokens)
	}
}

// ---- configFromRequest ------------------------------------------------------

func TestConfigFromRequest_NilRequest(t *testing.T) {
	cfg, err := configFromRequest(nil)
	if err != nil || cfg != nil {
		t.Fatalf("nil request: cfg=%v err=%v, want (nil, nil)", cfg, err)
	}
}

func TestConfigFromRequest_NilConfig(t *testing.T) {
	cfg, err := configFromRequest(&ai.ModelRequest{})
	if err != nil || cfg != nil {
		t.Fatalf("nil Config: cfg=%v err=%v, want (nil, nil)", cfg, err)
	}
}

func TestConfigFromRequest_GenerationCommonConfig(t *testing.T) {
	temp := 0.8

	t.Run("pointer form", func(t *testing.T) {
		cfg, err := configFromRequest(&ai.ModelRequest{
			Config: &ai.GenerationCommonConfig{
				MaxOutputTokens: 1024,
				Temperature:     temp,
				TopP:            0.95,
				StopSequences:   []string{"END"},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.MaxTokens != 1024 {
			t.Errorf("MaxTokens = %d, want 1024", cfg.MaxTokens)
		}
		if cfg.Temperature == nil || *cfg.Temperature != float32(temp) {
			t.Errorf("Temperature = %v, want %v", cfg.Temperature, temp)
		}
		if cfg.TopP == nil || *cfg.TopP != 0.95 {
			t.Errorf("TopP = %v, want 0.95", cfg.TopP)
		}
		if len(cfg.StopSequences) != 1 || cfg.StopSequences[0] != "END" {
			t.Errorf("StopSequences = %v, want [END]", cfg.StopSequences)
		}
	})

	t.Run("value form", func(t *testing.T) {
		cfg, err := configFromRequest(&ai.ModelRequest{
			Config: ai.GenerationCommonConfig{MaxOutputTokens: 512},
		})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.MaxTokens != 512 {
			t.Errorf("MaxTokens = %d, want 512", cfg.MaxTokens)
		}
	})

	t.Run("zero temperature is not set", func(t *testing.T) {
		cfg, err := configFromRequest(&ai.ModelRequest{
			Config: &ai.GenerationCommonConfig{MaxOutputTokens: 100},
		})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Temperature != nil {
			t.Errorf("Temperature = %v, want nil for zero value", cfg.Temperature)
		}
		if cfg.TopP != nil {
			t.Errorf("TopP = %v, want nil for zero value", cfg.TopP)
		}
	})
}

func TestConfigFromRequest_UnsupportedType(t *testing.T) {
	_, err := configFromRequest(&ai.ModelRequest{Config: "bad type"})
	if err == nil || !strings.Contains(err.Error(), "unexpected config type") {
		t.Fatalf("expected type error, got %v", err)
	}
}

// ---- buildConverseInput -----------------------------------------------------

func TestBuildConverseInput_NilRequest(t *testing.T) {
	b := &Bedrock{}
	_, err := b.buildConverseInput("model-id", nil)
	if err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil-request error, got %v", err)
	}
}

func TestBuildConverseInput_NonClaudeLeavesInferenceConfigUnset(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("amazon.nova-pro-v1:0", &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.InferenceConfig != nil {
		t.Fatalf("InferenceConfig = %v, want nil", out.InferenceConfig)
	}
}

func TestBuildConverseInput_ClaudeDefaultMaxTokens(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		want      int32
	}{
		{
			name:      "claude 3",
			modelName: "anthropic.claude-3-haiku-20240307-v1:0",
			want:      defaultClaudeMaxTokens,
		},
		{
			name:      "claude 3.5",
			modelName: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			want:      defaultExtendedClaudeMaxTokens,
		},
		{
			name:      "claude 4",
			modelName: "anthropic.claude-sonnet-4-20250514-v1:0",
			want:      defaultExtendedClaudeMaxTokens,
		},
	}

	b := &Bedrock{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := b.buildConverseInput(tt.modelName, &ai.ModelRequest{
				Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if out.InferenceConfig == nil || out.InferenceConfig.MaxTokens == nil {
				t.Fatalf("InferenceConfig.MaxTokens = nil, want %d", tt.want)
			}
			if *out.InferenceConfig.MaxTokens != tt.want {
				t.Fatalf("MaxTokens = %d, want %d", *out.InferenceConfig.MaxTokens, tt.want)
			}
		})
	}
}

func TestBuildConverseInput_ClaudePreservesConfiguredMaxTokens(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("anthropic.claude-3-5-sonnet-20241022-v2:0", &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}}},
		Config:   &Config{MaxTokens: 123},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.InferenceConfig == nil || out.InferenceConfig.MaxTokens == nil {
		t.Fatal("InferenceConfig.MaxTokens = nil, want 123")
	}
	if *out.InferenceConfig.MaxTokens != 123 {
		t.Fatalf("MaxTokens = %d, want 123", *out.InferenceConfig.MaxTokens)
	}
}

func TestBuildConverseInput_SystemTextPrompt(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("You are a helpful assistant.")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// System message must go to the System field, not Messages.
	if len(out.System) != 1 {
		t.Fatalf("len(System) = %d, want 1", len(out.System))
	}
	sysText, ok := out.System[0].(*types.SystemContentBlockMemberText)
	if !ok {
		t.Fatalf("System[0] type = %T, want *SystemContentBlockMemberText", out.System[0])
	}
	if sysText.Value != "You are a helpful assistant." {
		t.Errorf("System text = %q, want helpful assistant", sysText.Value)
	}
	// User message must be in Messages.
	if len(out.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1 (system excluded)", len(out.Messages))
	}
}

func TestBuildConverseInput_CachePointInSystem(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					ai.NewTextPart("big static prompt"),
					NewCachePointPart(),
				},
			},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("go")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.System) != 2 {
		t.Fatalf("len(System) = %d, want 2 (text + cache point)", len(out.System))
	}
	if _, ok := out.System[1].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Errorf("System[1] type = %T, want *SystemContentBlockMemberCachePoint", out.System[1])
	}
}

func TestBuildConverseInput_ToolDefinitions(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("use a tool")}},
		},
		Tools: []*ai.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Returns current weather",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig == nil || len(out.ToolConfig.Tools) != 1 {
		t.Fatalf("ToolConfig = %v, want 1 tool", out.ToolConfig)
	}
	spec, ok := out.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("tool type = %T, want *ToolMemberToolSpec", out.ToolConfig.Tools[0])
	}
	if aws.ToString(spec.Value.Name) != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", aws.ToString(spec.Value.Name))
	}
	if aws.ToString(spec.Value.Description) != "Returns current weather" {
		t.Errorf("tool desc = %q, want Returns current weather", aws.ToString(spec.Value.Description))
	}
}

func TestBuildConverseInput_ToolRequestPart(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("weather?")}},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{
						Name:  "get_weather",
						Ref:   "call-1",
						Input: map[string]any{"location": "Paris"},
					}),
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(out.Messages))
	}
	if len(out.Messages[1].Content) != 1 {
		t.Fatalf("assistant message has %d blocks, want 1", len(out.Messages[1].Content))
	}
	toolUse, ok := out.Messages[1].Content[0].(*types.ContentBlockMemberToolUse)
	if !ok {
		t.Fatalf("block type = %T, want *ContentBlockMemberToolUse", out.Messages[1].Content[0])
	}
	if aws.ToString(toolUse.Value.Name) != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", aws.ToString(toolUse.Value.Name))
	}
	if aws.ToString(toolUse.Value.ToolUseId) != "call-1" {
		t.Errorf("tool use id = %q, want call-1", aws.ToString(toolUse.Value.ToolUseId))
	}
}

func TestBuildConverseInput_ToolResultPart(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("weather?")}},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{
						Name: "get_weather", Ref: "call-1",
						Input: map[string]any{"location": "Paris"},
					}),
				},
			},
			{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name:   "get_weather",
						Ref:    "call-1",
						Output: "sunny, 25°C",
					}),
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// RoleTool messages go into Messages as a user-turn tool result.
	toolTurn := out.Messages[2]
	if len(toolTurn.Content) != 1 {
		t.Fatalf("tool turn has %d blocks, want 1", len(toolTurn.Content))
	}
	result, ok := toolTurn.Content[0].(*types.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("block type = %T, want *ContentBlockMemberToolResult", toolTurn.Content[0])
	}
	if aws.ToString(result.Value.ToolUseId) != "call-1" {
		t.Errorf("tool use id = %q, want call-1", aws.ToString(result.Value.ToolUseId))
	}
	if result.Value.Status != types.ToolResultStatusSuccess {
		t.Errorf("status = %v, want success", result.Value.Status)
	}
}

func TestBuildConverseInput_AssistantRemovedWhenToolsPresent(t *testing.T) {
	// Bedrock rejects conversations that end with an assistant message when tools are configured.
	// The plugin removes the trailing assistant message automatically.
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hi")}},
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hello")}},
		},
		Tools: []*ai.ToolDefinition{{Name: "noop", Description: "does nothing"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1 (trailing assistant removed)", len(out.Messages))
	}
	if out.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("remaining message role = %v, want user", out.Messages[0].Role)
	}
}

func TestBuildConverseInput_MultiTurnText(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("Hello")}},
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("Hi there")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("How are you?")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 3 {
		t.Fatalf("len(Messages) = %d, want 3", len(out.Messages))
	}
	if out.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("msg[0] role = %v, want user", out.Messages[0].Role)
	}
	if out.Messages[1].Role != types.ConversationRoleAssistant {
		t.Errorf("msg[1] role = %v, want assistant", out.Messages[1].Role)
	}
}

func TestBuildConverseInput_SkipsNilMessagesAndParts(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			nil,
			{Role: ai.RoleUser, Content: []*ai.Part{nil, ai.NewTextPart("hi")}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(out.Messages))
	}
	if len(out.Messages[0].Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(out.Messages[0].Content))
	}
}

func TestMediaToBlock_ImageAndDocument(t *testing.T) {
	payload := []byte("file bytes")
	encoded := base64.StdEncoding.EncodeToString(payload)

	imageBlock, err := mediaToBlock(ai.NewMediaPart(" Image/JPEG ; charset=binary ", "\n data:image/jpeg;base64,"+encoded+" \t"))
	if err != nil {
		t.Fatal(err)
	}
	image, ok := imageBlock.(*types.ContentBlockMemberImage)
	if !ok {
		t.Fatalf("image block = %T, want *ContentBlockMemberImage", imageBlock)
	}
	if image.Value.Format != types.ImageFormatJpeg {
		t.Errorf("image format = %q, want jpeg", image.Value.Format)
	}
	imageBytes, ok := image.Value.Source.(*types.ImageSourceMemberBytes)
	if !ok {
		t.Fatalf("image source = %T, want bytes", image.Value.Source)
	}
	if string(imageBytes.Value) != string(payload) {
		t.Errorf("image bytes = %q, want %q", string(imageBytes.Value), string(payload))
	}

	docBlock, err := mediaToBlock(ai.NewMediaPart("text/markdown", "\n "+encoded+"\t "))
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := docBlock.(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("document block = %T, want *ContentBlockMemberDocument", docBlock)
	}
	if doc.Value.Format != types.DocumentFormatMd {
		t.Errorf("document format = %q, want md", doc.Value.Format)
	}
}

func TestMediaToBlock_StrictValidation(t *testing.T) {
	tests := []struct {
		name        string
		part        *ai.Part
		wantMessage string
	}{
		{
			name:        "remote URL",
			part:        ai.NewMediaPart("image/png", "https://example.com/cat.png"),
			wantMessage: "remote URLs are not supported",
		},
		{
			name:        "raw content",
			part:        ai.NewMediaPart("image/png", "not base64"),
			wantMessage: "decode base64 media",
		},
		{
			name:        "missing MIME",
			part:        ai.NewMediaPart("", base64.StdEncoding.EncodeToString([]byte("bytes"))),
			wantMessage: "no content type",
		},
		{
			name:        "unknown MIME",
			part:        ai.NewMediaPart("application/zip", base64.StdEncoding.EncodeToString([]byte("bytes"))),
			wantMessage: "unsupported media MIME type",
		},
		{
			name:        "malformed data URL",
			part:        ai.NewMediaPart("image/png", "data:image/png,not-base64"),
			wantMessage: "data URL must be base64-encoded",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := mediaToBlock(tt.part)
			if err == nil || !strings.Contains(err.Error(), tt.wantMessage) {
				t.Fatalf("mediaToBlock() error = %v, want containing %q", err, tt.wantMessage)
			}
		})
	}
}

// ---- convertResponse --------------------------------------------------------

func TestConvertResponse_ToolUseBlock(t *testing.T) {
	b := &Bedrock{}
	inputDoc := document.NewLazyDocument(map[string]any{"location": "Paris"})

	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String("call-123"),
							Name:      aws.String("get_weather"),
							Input:     inputDoc,
						},
					},
				},
			},
		},
		StopReason: types.StopReasonToolUse,
	}

	req := &ai.ModelRequest{
		Tools: []*ai.ToolDefinition{
			{
				Name: "get_weather",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"location": map[string]any{"type": "string"}},
				},
			},
		},
	}
	got, err := b.convertResponse(resp, req)
	if err != nil {
		t.Fatal(err)
	}

	if got.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop (tool use maps to Stop)", got.FinishReason)
	}
	if len(got.Message.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1", len(got.Message.Content))
	}
	if !got.Message.Content[0].IsToolRequest() {
		t.Fatalf("part kind = %v, want tool request", got.Message.Content[0].Kind)
	}
	tr := got.Message.Content[0].ToolRequest
	if tr.Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", tr.Name)
	}
	if tr.Ref != "call-123" {
		t.Errorf("tool ref = %q, want call-123", tr.Ref)
	}
}

func TestConvertResponse_ToolUsePreservesLargeIntegerInput(t *testing.T) {
	b := &Bedrock{}
	const largeID int64 = 9007199254740993
	inputDoc := document.NewLazyDocument(map[string]any{"id": largeID})

	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String("call-123"),
							Name:      aws.String("lookup"),
							Input:     inputDoc,
						},
					},
				},
			},
		},
		StopReason: types.StopReasonToolUse,
	}
	req := &ai.ModelRequest{
		Tools: []*ai.ToolDefinition{
			{
				Name: "lookup",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"id": map[string]any{"type": "integer"}},
				},
			},
		},
	}

	got, err := b.convertResponse(resp, req)
	if err != nil {
		t.Fatal(err)
	}
	input, ok := got.Message.Content[0].ToolRequest.Input.(map[string]any)
	if !ok {
		t.Fatalf("tool input = %T, want map[string]any", got.Message.Content[0].ToolRequest.Input)
	}
	if input["id"] != largeID {
		t.Fatalf("id = %#v (%T), want %d", input["id"], input["id"], largeID)
	}
}

func TestConvertResponse_ToolUseIntegerSchemaTruncatesDecimalJSONNumber(t *testing.T) {
	b := &Bedrock{}
	inputDoc := document.NewLazyDocument(map[string]any{"count": 7.9})

	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String("call-123"),
							Name:      aws.String("count_items"),
							Input:     inputDoc,
						},
					},
				},
			},
		},
		StopReason: types.StopReasonToolUse,
	}
	req := &ai.ModelRequest{
		Tools: []*ai.ToolDefinition{
			{
				Name: "count_items",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"count": map[string]any{"type": "integer"}},
				},
			},
		},
	}

	got, err := b.convertResponse(resp, req)
	if err != nil {
		t.Fatal(err)
	}
	input, ok := got.Message.Content[0].ToolRequest.Input.(map[string]any)
	if !ok {
		t.Fatalf("tool input = %T, want map[string]any", got.Message.Content[0].ToolRequest.Input)
	}
	if input["count"] != int64(7) {
		t.Fatalf("count = %#v (%T), want int64(7)", input["count"], input["count"])
	}
}

func TestConvertResponse_TokenUsage(t *testing.T) {
	b := &Bedrock{}
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "hi"},
				},
			},
		},
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:          aws.Int32(100),
			OutputTokens:         aws.Int32(50),
			TotalTokens:          aws.Int32(150),
			CacheReadInputTokens: aws.Int32(20),
		},
	}

	got, err := b.convertResponse(resp, &ai.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Usage == nil {
		t.Fatal("expected Usage to be set")
	}
	if got.Usage.InputTokens != 100 {
		t.Errorf("InputTokens = %d, want 100", got.Usage.InputTokens)
	}
	if got.Usage.OutputTokens != 50 {
		t.Errorf("OutputTokens = %d, want 50", got.Usage.OutputTokens)
	}
	if got.Usage.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150", got.Usage.TotalTokens)
	}
	if got.Usage.CachedContentTokens != 20 {
		t.Errorf("CachedContentTokens = %d, want 20", got.Usage.CachedContentTokens)
	}
}

func TestConvertResponse_SetsOriginalRequest(t *testing.T) {
	b := &Bedrock{}
	req := &ai.ModelRequest{}
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}}},
		},
		StopReason: types.StopReasonEndTurn,
	}

	got, err := b.convertResponse(resp, req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Request != req {
		t.Fatalf("Request = %p, want %p", got.Request, req)
	}
}

func TestConvertResponse_NilOutputPlaceholder(t *testing.T) {
	b := &Bedrock{}
	got, err := b.convertResponse(&bedrockruntime.ConverseOutput{
		StopReason: types.StopReasonGuardrailIntervened,
	}, &ai.ModelRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.FinishReason != ai.FinishReasonBlocked {
		t.Fatalf("FinishReason = %v, want blocked", got.FinishReason)
	}
	if got.Message == nil || got.Message.Role != ai.RoleModel {
		t.Fatalf("Message = %#v, want model message", got.Message)
	}
	if len(got.Message.Content) != 1 {
		t.Fatalf("len(Content) = %d, want 1 placeholder", len(got.Message.Content))
	}
	if !got.Message.Content[0].IsText() || got.Message.Content[0].Text != "" {
		t.Errorf("placeholder part = %#v, want empty text", got.Message.Content[0])
	}
}

func TestConvertResponse_UnknownContentVariant(t *testing.T) {
	b := &Bedrock{}
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberImage{
						Value: types.ImageBlock{
							Format: types.ImageFormatPng,
							Source: &types.ImageSourceMemberBytes{Value: []byte("png")},
						},
					},
				},
			},
		},
	}
	_, err := b.convertResponse(resp, &ai.ModelRequest{})
	if err == nil || !strings.Contains(err.Error(), "unhandled response content variant") {
		t.Fatalf("expected unhandled content error, got %v", err)
	}
}

func TestConvertResponse_StopReasonMappedToFinishReason(t *testing.T) {
	b := &Bedrock{}
	makeResp := func(sr types.StopReason) *bedrockruntime.ConverseOutput {
		return &bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "x"}}},
			},
			StopReason: sr,
		}
	}

	tests := []struct {
		sr   types.StopReason
		want ai.FinishReason
	}{
		{types.StopReasonEndTurn, ai.FinishReasonStop},
		{types.StopReasonMaxTokens, ai.FinishReasonLength},
		{types.StopReasonModelContextWindowExceeded, ai.FinishReasonLength},
		{types.StopReasonContentFiltered, ai.FinishReasonBlocked},
		{types.StopReasonGuardrailIntervened, ai.FinishReasonBlocked},
		{types.StopReasonMalformedModelOutput, ai.FinishReasonOther},
		{types.StopReasonMalformedToolUse, ai.FinishReasonOther},
	}
	for _, tt := range tests {
		got, err := b.convertResponse(makeResp(tt.sr), &ai.ModelRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if got.FinishReason != tt.want {
			t.Errorf("stop=%q: FinishReason=%q, want %q", tt.sr, got.FinishReason, tt.want)
		}
	}
}

// ---- types helpers ----------------------------------------------------------

func TestMetadataBytes_TypeAssertions(t *testing.T) {
	tests := []struct {
		name  string
		val   any
		want  string
		empty bool
	}{
		{name: "[]byte value", val: []byte("hello"), want: "hello"},
		{name: "base64 string", val: "aGVsbG8=", want: "hello"},
		{name: "nil map", val: nil, empty: true},
		{name: "unsupported type", val: 42, empty: true},
		{name: "invalid base64 string", val: "not-base64!!!", empty: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var meta map[string]any
			if tt.val != nil {
				meta = map[string]any{"key": tt.val}
			}
			got := metadataBytes(meta, "key")
			if tt.empty {
				if len(got) != 0 {
					t.Errorf("expected empty, got %q", got)
				}
			} else {
				if string(got) != tt.want {
					t.Errorf("got %q, want %q", string(got), tt.want)
				}
			}
		})
	}
}

func TestMetadataBytes_NilMap(t *testing.T) {
	if got := metadataBytes(nil, "any"); len(got) != 0 {
		t.Errorf("nil map should return nil, got %q", got)
	}
}

func TestCachePointType_PresentAndAbsent(t *testing.T) {
	p := NewCachePointPart()
	cpt, ok := CachePointType(p)
	if !ok {
		t.Fatal("expected CachePointType to be present")
	}
	if cpt != types.CachePointTypeDefault {
		t.Errorf("CachePointType = %v, want Default", cpt)
	}

	// Non-cache-point part returns false.
	_, ok = CachePointType(ai.NewTextPart("hello"))
	if ok {
		t.Error("text part should not have a CachePointType")
	}
}

// ---- schema helpers ---------------------------------------------------------

func TestNewObjectSchema(t *testing.T) {
	props := map[string]any{
		"name": map[string]any{"type": "string"},
	}
	schema := NewObjectSchema(props, []string{"name"})

	if schema["type"] != "object" {
		t.Errorf("type = %v, want object", schema["type"])
	}
	if _, ok := schema["properties"]; !ok {
		t.Error("properties field missing")
	}
	req, _ := schema["required"].([]string)
	if len(req) != 1 || req[0] != "name" {
		t.Errorf("required = %v, want [name]", req)
	}

	// Without required fields.
	schema2 := NewObjectSchema(props, nil)
	if _, ok := schema2["required"]; ok {
		t.Error("required field should be absent when no required properties")
	}
}

func TestNewStringSchema(t *testing.T) {
	schema := NewStringSchema("A label", []string{"a", "b"})
	if schema["type"] != "string" {
		t.Errorf("type = %v, want string", schema["type"])
	}
	if schema["description"] != "A label" {
		t.Errorf("description = %v, want 'A label'", schema["description"])
	}
	enum, _ := schema["enum"].([]string)
	if len(enum) != 2 {
		t.Errorf("enum = %v, want [a b]", enum)
	}

	// No description, no enum.
	schema2 := NewStringSchema("", nil)
	if _, ok := schema2["description"]; ok {
		t.Error("description should be absent when empty")
	}
	if _, ok := schema2["enum"]; ok {
		t.Error("enum should be absent when nil")
	}
}

func TestNewNumberSchema(t *testing.T) {
	min, max := 0.0, 100.0
	schema := NewNumberSchema("score", &min, &max)
	if schema["type"] != "number" {
		t.Errorf("type = %v, want number", schema["type"])
	}
	if schema["minimum"] != min {
		t.Errorf("minimum = %v, want 0", schema["minimum"])
	}
	if schema["maximum"] != max {
		t.Errorf("maximum = %v, want 100", schema["maximum"])
	}

	// Nil bounds.
	schema2 := NewNumberSchema("", nil, nil)
	if _, ok := schema2["minimum"]; ok {
		t.Error("minimum should be absent when nil")
	}
}

func TestNewArraySchema(t *testing.T) {
	itemSchema := map[string]any{"type": "string"}
	schema := NewArraySchema(itemSchema, "list of strings")
	if schema["type"] != "array" {
		t.Errorf("type = %v, want array", schema["type"])
	}
	if schema["description"] != "list of strings" {
		t.Errorf("description = %v, want 'list of strings'", schema["description"])
	}
	if schema["items"] == nil {
		t.Error("items field missing")
	}
}

// ---- normalizeSchema --------------------------------------------------------

func TestNormalizeSchema_StringInput(t *testing.T) {
	b := &Bedrock{}
	input := `{"type":"object","properties":{"x":{"type":"integer"}}}`
	result, err := b.normalizeSchema(input)
	if err != nil {
		t.Fatalf("normalizeSchema string: %v", err)
	}
	if result["type"] != "object" {
		t.Errorf("type = %v, want object", result["type"])
	}
}

func TestNormalizeSchema_BytesInput(t *testing.T) {
	b := &Bedrock{}
	input := []byte(`{"type":"object"}`)
	result, err := b.normalizeSchema(input)
	if err != nil {
		t.Fatalf("normalizeSchema bytes: %v", err)
	}
	if result["type"] != "object" {
		t.Errorf("type = %v, want object", result["type"])
	}
}

func TestNormalizeSchema_InvalidJSON(t *testing.T) {
	b := &Bedrock{}
	_, err := b.normalizeSchema("not json")
	if err == nil {
		t.Fatal("expected error for invalid JSON string")
	}
}

// ---- convertValueWithSchema -------------------------------------------------

func TestConvertValueWithSchema_NumericConversions(t *testing.T) {
	b := &Bedrock{}

	tests := []struct {
		name     string
		value    any
		schema   map[string]any
		wantType string
	}{
		{"int to float64", int(5), map[string]any{"type": "number"}, "float64"},
		{"int32 to float64", int32(5), map[string]any{"type": "number"}, "float64"},
		{"int64 to float64", int64(5), map[string]any{"type": "number"}, "float64"},
		{"float32 to float64", float32(5.5), map[string]any{"type": "number"}, "float64"},
		{"int to int64", int(5), map[string]any{"type": "integer"}, "int64"},
		{"float64 to int64", float64(7.9), map[string]any{"type": "integer"}, "int64"},
		{"string to float64", "3.14", map[string]any{"type": "number"}, "float64"},
		{"string to int64", "42", map[string]any{"type": "integer"}, "int64"},
		{"string to bool", "true", map[string]any{"type": "boolean"}, "bool"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := b.convertValueWithSchema(tt.value, tt.schema)
			switch tt.wantType {
			case "float64":
				if _, ok := got.(float64); !ok {
					t.Errorf("got type %T, want float64", got)
				}
			case "int64":
				if _, ok := got.(int64); !ok {
					t.Errorf("got type %T, want int64", got)
				}
			case "bool":
				if _, ok := got.(bool); !ok {
					t.Errorf("got type %T, want bool", got)
				}
			}
		})
	}
}

func TestConvertValueWithSchema_JSONNumberToString(t *testing.T) {
	b := &Bedrock{}
	got := b.convertValueWithSchema(json.Number("123"), map[string]any{"type": "string"})
	if got != "123" {
		t.Fatalf("got %#v (%T), want string 123", got, got)
	}
}

func TestConvertValueWithSchema_ArrayConversion(t *testing.T) {
	b := &Bedrock{}
	arr := []any{int(1), int(2), int(3)}
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{"type": "number"},
	}
	got := b.convertValueWithSchema(arr, schema)
	result, ok := got.([]any)
	if !ok {
		t.Fatalf("got type %T, want []any", got)
	}
	for i, v := range result {
		if _, ok := v.(float64); !ok {
			t.Errorf("result[%d] type = %T, want float64", i, v)
		}
	}
}

func TestConvertValueWithSchema_NilSchema(t *testing.T) {
	b := &Bedrock{}
	// Nil schema — value should be returned unchanged.
	got := b.convertValueWithSchema("hello", nil)
	if got != "hello" {
		t.Errorf("got %v, want hello", got)
	}
}

// ---- convertMapWithSchema ---------------------------------------------------

func TestConvertMapWithSchema_ObjectSchema(t *testing.T) {
	b := &Bedrock{}
	inputMap := map[string]any{
		"count": int(3),
		"label": "hello",
	}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "number"},
			"label": map[string]any{"type": "string"},
		},
	}
	got := b.convertMapWithSchema(inputMap, schema)
	result, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got type %T, want map[string]any", got)
	}
	if _, ok := result["count"].(float64); !ok {
		t.Errorf("count type = %T, want float64", result["count"])
	}
	if result["label"] != "hello" {
		t.Errorf("label = %v, want hello", result["label"])
	}
}

func TestConvertMapWithSchema_NilSchema(t *testing.T) {
	b := &Bedrock{}
	inputMap := map[string]any{"x": int(1)}
	got := b.convertMapWithSchema(inputMap, nil)
	result, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got type %T, want map[string]any", got)
	}
	if result["x"] != int(1) {
		t.Errorf("x = %v, want 1 (unchanged)", result["x"])
	}
}

// ---- convertToolInputTypes --------------------------------------------------

func TestConvertToolInputTypes_NoMatchingTool(t *testing.T) {
	b := &Bedrock{}
	inputMap := map[string]any{"n": int(5)}
	tools := []*ai.ToolDefinition{{Name: "other_tool", InputSchema: map[string]any{"type": "object"}}}
	got := b.convertToolInputTypes(inputMap, "missing_tool", tools)
	result, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got type %T, want map[string]any", got)
	}
	if result["n"] != int(5) {
		t.Errorf("n = %v, want 5 (unchanged)", result["n"])
	}
}

func TestConvertToolInputTypes_ConvertsMatchingTool(t *testing.T) {
	b := &Bedrock{}
	inputMap := map[string]any{"price": int(42)}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"price": map[string]any{"type": "number"},
		},
	}
	tools := []*ai.ToolDefinition{{Name: "get_price", InputSchema: schema}}
	got := b.convertToolInputTypes(inputMap, "get_price", tools)
	result, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got type %T, want map[string]any", got)
	}
	if _, ok := result["price"].(float64); !ok {
		t.Errorf("price type = %T, want float64 after conversion", result["price"])
	}
}

// ---- generateTextSync (integration via mock HTTP server) --------------------

// TestGenerateTextSync_BasicRoundTrip exercises the full sync generation path
// through a mock Bedrock Converse endpoint. It verifies that the plugin builds
// a correct request body, that InferenceConfig is forwarded, and that the text
// response and stop reason are mapped back to Genkit types.
func TestGenerateTextSync_BasicRoundTrip(t *testing.T) {
	// The AWS SDK sends Converse requests as JSON-over-HTTP, so we can intercept.
	var gotBody map[string]json.RawMessage
	var gotPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		// Minimal valid ConverseOutput shape.
		_, _ = fmt.Fprint(w, `{
			"output":     {"message":{"role":"assistant","content":[{"text":"pong"}]}},
			"stopReason": "end_turn",
			"usage":      {"inputTokens":5,"outputTokens":2,"totalTokens":7}
		}`)
	}))
	defer server.Close()

	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:       "us-east-1",
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		HTTPClient:   server.Client(),
		BaseEndpoint: aws.String(server.URL),
	})
	b := &Bedrock{client: client, initted: true}

	resp, err := b.generateText(context.Background(), "anthropic.claude-3-haiku", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("ping")}},
		},
		Config: &Config{MaxTokens: 100},
	}, nil)
	if err != nil {
		t.Fatalf("generateText error: %v", err)
	}

	// Path must be the Converse endpoint.
	if !strings.Contains(gotPath, "/converse") {
		t.Errorf("path = %q, want converse endpoint", gotPath)
	}
	// InferenceConfig should have been forwarded.
	if gotBody["inferenceConfig"] == nil {
		t.Error("inferenceConfig missing from request body")
	}
	// Response text should be "pong".
	if resp.Text() != "pong" {
		t.Errorf("Text() = %q, want pong", resp.Text())
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want Stop", resp.FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 7 {
		t.Errorf("Usage = %v, want TotalTokens=7", resp.Usage)
	}
}

// ---- buildConverseInput ToolChoice wiring -----------------------------------

func toolReq() *ai.ModelRequest {
	return &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("use a tool")}},
		},
		Tools: []*ai.ToolDefinition{
			{Name: "get_weather", Description: "weather", InputSchema: map[string]any{"type": "object"}},
		},
	}
}

func TestBuildConverseInput_ToolChoiceAuto(t *testing.T) {
	b := &Bedrock{}
	req := toolReq()
	req.Config = &Config{ToolChoice: ToolChoiceAuto}
	out, err := b.buildConverseInput("model-id", req)
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig == nil {
		t.Fatal("ToolConfig is nil")
	}
	if _, ok := out.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); !ok {
		t.Errorf("ToolChoice type = %T, want *ToolChoiceMemberAuto", out.ToolConfig.ToolChoice)
	}
}

func TestBuildConverseInput_ToolChoiceRequired(t *testing.T) {
	b := &Bedrock{}
	req := toolReq()
	req.Config = &Config{ToolChoice: ToolChoiceRequired}
	out, err := b.buildConverseInput("model-id", req)
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig == nil {
		t.Fatal("ToolConfig is nil")
	}
	if _, ok := out.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAny); !ok {
		t.Errorf("ToolChoice type = %T, want *ToolChoiceMemberAny", out.ToolConfig.ToolChoice)
	}
}

func TestBuildConverseInput_ToolChoiceNamedTool(t *testing.T) {
	b := &Bedrock{}
	req := toolReq()
	req.Config = &Config{ToolChoice: "get_weather"}
	out, err := b.buildConverseInput("model-id", req)
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig == nil {
		t.Fatal("ToolConfig is nil")
	}
	specific, ok := out.ToolConfig.ToolChoice.(*types.ToolChoiceMemberTool)
	if !ok {
		t.Fatalf("ToolChoice type = %T, want *ToolChoiceMemberTool", out.ToolConfig.ToolChoice)
	}
	if aws.ToString(specific.Value.Name) != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", aws.ToString(specific.Value.Name))
	}
}

func TestBuildConverseInput_ToolChoiceIgnoredWithoutTools(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hello")}},
		},
		Config: &Config{ToolChoice: ToolChoiceAuto},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig != nil {
		t.Errorf("ToolConfig = %v, want nil when no tools provided", out.ToolConfig)
	}
}

func TestBuildConverseInput_ToolChoiceNone(t *testing.T) {
	b := &Bedrock{}
	req := toolReq()
	req.Config = &Config{ToolChoice: ToolChoiceNone}
	out, err := b.buildConverseInput("model-id", req)
	if err != nil {
		t.Fatal(err)
	}
	if out.ToolConfig != nil {
		t.Errorf("ToolConfig = %v, want nil for ToolChoiceNone", out.ToolConfig)
	}
}

func TestBuildConverseInput_ToolChoiceNoneSkipsInvalidTool(t *testing.T) {
	b := &Bedrock{}
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hello")}},
		},
		Tools:  []*ai.ToolDefinition{nil},
		Config: &Config{ToolChoice: ToolChoiceNone},
	}

	out, err := b.buildConverseInput("model-id", req)
	if err != nil {
		t.Fatalf("buildConverseInput() error = %v", err)
	}
	if out.ToolConfig != nil {
		t.Errorf("ToolConfig = %v, want nil for ToolChoiceNone", out.ToolConfig)
	}
}

func TestBuildConverseInput_ToolSchemaConversionFailureFallsBack(t *testing.T) {
	b := &Bedrock{}
	out, err := b.buildConverseInput("model-id", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("use a tool")}},
		},
		Tools: []*ai.ToolDefinition{
			{
				Name:        "bad_schema_tool",
				Description: "schema cannot be JSON marshaled",
				InputSchema: map[string]any{"bad": func() {}},
			},
		},
	})
	if err != nil {
		t.Fatalf("buildConverseInput() error = %v", err)
	}
	if out.ToolConfig == nil || len(out.ToolConfig.Tools) != 1 {
		t.Fatalf("ToolConfig = %v, want one tool", out.ToolConfig)
	}
	spec, ok := out.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("tool type = %T, want *ToolMemberToolSpec", out.ToolConfig.Tools[0])
	}
	if spec.Value.InputSchema != nil {
		t.Errorf("InputSchema = %T, want nil fallback after conversion failure", spec.Value.InputSchema)
	}
}
