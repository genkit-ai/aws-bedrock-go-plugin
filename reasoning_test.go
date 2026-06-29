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
	"encoding/base64"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
)

// --- Request round-trip -----------------------------------------------------

func TestReasoningPartToContentBlocks_RoundTripsBedrockReasoning(t *testing.T) {
	// Signed reasoning text → ReasoningContentBlockMemberReasoningText.
	signed := newBedrockReasoningPart("signed thought", "sig", nil)
	blocks := reasoningPartToContentBlocks(signed)
	if len(blocks) != 1 {
		t.Fatalf("len(blocks) = %d, want 1", len(blocks))
	}
	rc, ok := blocks[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("blocks[0] = %T, want reasoning content", blocks[0])
	}
	text, ok := rc.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok {
		t.Fatalf("blocks[0].Value = %T, want reasoning text", rc.Value)
	}
	if aws.ToString(text.Value.Text) != "signed thought" {
		t.Errorf("text = %q, want signed thought", aws.ToString(text.Value.Text))
	}
	if aws.ToString(text.Value.Signature) != "sig" {
		t.Errorf("signature = %q, want sig", aws.ToString(text.Value.Signature))
	}

	// Redacted reasoning stored as a base64 string (JSON round-trip shape) →
	// ReasoningContentBlockMemberRedactedContent with the decoded bytes.
	redacted := ai.NewReasoningPart("", nil)
	redacted.Metadata[redactedReasoningMetadataKey] = base64.StdEncoding.EncodeToString([]byte("encrypted"))
	blocks = reasoningPartToContentBlocks(redacted)
	if len(blocks) != 1 {
		t.Fatalf("redacted len(blocks) = %d, want 1", len(blocks))
	}
	rc, ok = blocks[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("redacted blocks[0] = %T, want reasoning content", blocks[0])
	}
	red, ok := rc.Value.(*types.ReasoningContentBlockMemberRedactedContent)
	if !ok {
		t.Fatalf("redacted blocks[0].Value = %T, want redacted content", rc.Value)
	}
	if string(red.Value) != "encrypted" {
		t.Errorf("redacted = %q, want encrypted", string(red.Value))
	}
}

func TestReasoningPartToContentBlocks_SkipsGenericReasoning(t *testing.T) {
	// A generic reasoning part (signature under the framework "signature" key,
	// not the Bedrock-specific key) must not round-trip into request blocks.
	p := ai.NewReasoningPart("signed elsewhere", []byte("foreign-sig"))
	if blocks := reasoningPartToContentBlocks(p); len(blocks) != 0 {
		t.Fatalf("len(blocks) = %d, want 0", len(blocks))
	}
}

func TestBuildConverseInput_SkipsGenericReasoning(t *testing.T) {
	b := &Bedrock{}
	input := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewTextPart("question"),
					ai.NewReasoningPart("internal monologue from a prior turn", nil),
					ai.NewTextPart("more question"),
				},
			},
		},
	}

	out, err := b.buildConverseInput("anthropic.claude-3-sonnet", input)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(out.Messages))
	}
	blocks := out.Messages[0].Content
	if len(blocks) != 2 {
		t.Fatalf("len(blocks) = %d, want 2 (no reasoning leakage)", len(blocks))
	}
	for i, blk := range blocks {
		text, ok := blk.(*types.ContentBlockMemberText)
		if !ok {
			t.Fatalf("blocks[%d] = %T, want text", i, blk)
		}
		if text.Value == "internal monologue from a prior turn" {
			t.Errorf("reasoning text leaked into block %d", i)
		}
	}
}

// --- Response parse ---------------------------------------------------------

func TestConvertResponse_ReasoningSignatureAndRedacted(t *testing.T) {
	b := &Bedrock{}
	redacted := []byte("encrypted")
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberReasoningContent{
						Value: &types.ReasoningContentBlockMemberReasoningText{
							Value: types.ReasoningTextBlock{
								Text:      aws.String("thinking"),
								Signature: aws.String("sig"),
							},
						},
					},
					&types.ContentBlockMemberReasoningContent{
						Value: &types.ReasoningContentBlockMemberRedactedContent{Value: redacted},
					},
				},
			},
		},
	}

	got, err := b.convertResponse(resp, &ai.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	parts := got.Message.Content
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}

	if !parts[0].IsReasoning() || parts[0].Text != "thinking" {
		t.Fatalf("parts[0] = %+v, want reasoning text", parts[0])
	}
	if sig, ok := parts[0].Metadata["signature"].([]byte); !ok || string(sig) != "sig" {
		t.Errorf("generic signature = %v, want sig", parts[0].Metadata["signature"])
	}
	if sig, ok := parts[0].Metadata[reasoningSignatureMetadataKey].([]byte); !ok || string(sig) != "sig" {
		t.Errorf("bedrock signature = %v, want sig", parts[0].Metadata[reasoningSignatureMetadataKey])
	}

	if !parts[1].IsReasoning() {
		t.Fatalf("parts[1] kind = %v, want reasoning", parts[1].Kind)
	}
	if red, ok := parts[1].Metadata[redactedReasoningMetadataKey].([]byte); !ok || string(red) != string(redacted) {
		t.Errorf("redacted = %v, want %q", parts[1].Metadata[redactedReasoningMetadataKey], string(redacted))
	}
}

// TestConvertResponse_TextSkipsReasoning is the proposal's sanity check: a
// response carrying both reasoning and text must expose only the text via
// Text(), so existing callers don't suddenly see thinking content.
func TestConvertResponse_TextSkipsReasoning(t *testing.T) {
	b := &Bedrock{}
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberReasoningContent{
						Value: &types.ReasoningContentBlockMemberReasoningText{
							Value: types.ReasoningTextBlock{
								Text:      aws.String("let me think about this"),
								Signature: aws.String("sig"),
							},
						},
					},
					&types.ContentBlockMemberText{Value: "the answer is 42"},
				},
			},
		},
	}

	got, err := b.convertResponse(resp, &ai.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if text := got.Text(); text != "the answer is 42" {
		t.Errorf("Text() = %q, want %q (reasoning leaked)", text, "the answer is 42")
	}
}

// --- Streaming --------------------------------------------------------------

func TestAppendReasoningDelta(t *testing.T) {
	acc := &streamAccumulator{}

	part, err := appendReasoningDelta(acc, &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"})
	if err != nil {
		t.Fatal(err)
	}
	if part == nil || !part.IsReasoning() || part.Text != "thinking" {
		t.Fatalf("text delta part = %+v, want reasoning %q", part, "thinking")
	}
	if got := acc.reasoning.String(); got != "thinking" {
		t.Errorf("accumulated reasoning = %q, want thinking", got)
	}

	part, err = appendReasoningDelta(acc, &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig"})
	if err != nil {
		t.Fatal(err)
	}
	if part != nil {
		t.Fatalf("signature delta returned part %v, want nil", part)
	}
	if acc.reasoningSignature != "sig" {
		t.Errorf("signature = %q, want sig", acc.reasoningSignature)
	}

	part, err = appendReasoningDelta(acc, &types.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("encrypted")})
	if err != nil {
		t.Fatal(err)
	}
	if part != nil {
		t.Fatalf("redacted delta returned part %v, want nil", part)
	}
	if string(acc.redactedReasoning) != "encrypted" {
		t.Errorf("redacted = %q, want encrypted", string(acc.redactedReasoning))
	}
}

func TestAppendReasoningDelta_UnknownErrors(t *testing.T) {
	if _, err := appendReasoningDelta(&streamAccumulator{}, &types.UnknownUnionMember{Tag: "future_reasoning_delta"}); err == nil {
		t.Fatal("expected error for unknown reasoning delta")
	}
}

func TestStreamFinalContent_ReasoningReassembly(t *testing.T) {
	acc := &streamAccumulator{reasoningSignature: "sig", redactedReasoning: []byte("encrypted")}
	acc.reasoning.WriteString("First thought. ")
	acc.reasoning.WriteString("Second thought.")
	acc.text.WriteString("Final answer.")

	parts := acc.finalContent()
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}

	// Reasoning precedes text so the assistant turn replays in order.
	if !parts[0].IsReasoning() || parts[0].Text != "First thought. Second thought." {
		t.Fatalf("parts[0] = %+v, want assembled reasoning", parts[0])
	}
	if sig, ok := parts[0].Metadata["signature"].([]byte); !ok || string(sig) != "sig" {
		t.Errorf("generic signature = %v, want sig", parts[0].Metadata["signature"])
	}
	if sig, ok := parts[0].Metadata[reasoningSignatureMetadataKey].([]byte); !ok || string(sig) != "sig" {
		t.Errorf("bedrock signature = %v, want sig", parts[0].Metadata[reasoningSignatureMetadataKey])
	}
	if red, ok := parts[0].Metadata[redactedReasoningMetadataKey].([]byte); !ok || string(red) != "encrypted" {
		t.Errorf("redacted = %v, want encrypted", parts[0].Metadata[redactedReasoningMetadataKey])
	}
	if !parts[1].IsText() || parts[1].Text != "Final answer." {
		t.Errorf("parts[1] = %+v, want text Final answer.", parts[1])
	}
}

// --- Config decode ----------------------------------------------------------

func TestConfigFromRequest_TypedAndAdditionalFields(t *testing.T) {
	thinking := map[string]any{"type": "enabled", "budget_tokens": 5000}
	input := &ai.ModelRequest{Config: &Config{
		MaxTokens:                    8000,
		AdditionalModelRequestFields: map[string]any{"thinking": thinking},
	}}
	cfg, err := configFromRequest(input)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxTokens != 8000 {
		t.Errorf("MaxTokens = %d, want 8000", cfg.MaxTokens)
	}
	if cfg.AdditionalModelRequestFields["thinking"] == nil {
		t.Error("thinking field dropped")
	}
}

// TestConfigFromRequest_LegacyMapKeys guards backward compatibility: the
// historical map config (with maxOutputTokens) must still drive MaxTokens.
func TestConfigFromRequest_LegacyMapKeys(t *testing.T) {
	cases := []map[string]interface{}{
		{"maxOutputTokens": 1024, "temperature": 0.5},
		{"max_tokens": 1024},
		{"maxOutputTokens": float64(1024)}, // JSON-decoded shape
	}
	for _, m := range cases {
		cfg, err := configFromRequest(&ai.ModelRequest{Config: m})
		if err != nil {
			t.Fatalf("config %v: %v", m, err)
		}
		ic := buildInferenceConfig(cfg)
		if ic == nil || ic.MaxTokens == nil || *ic.MaxTokens != 1024 {
			t.Errorf("config %v: MaxTokens = %v, want 1024", m, ic.MaxTokens)
		}
	}
}
