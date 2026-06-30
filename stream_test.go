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
	"errors"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
)

func TestConsumeStreamEvents_TextOnly(t *testing.T) {
	events := streamEvents(
		textDelta(0, "hel"),
		textDelta(0, "lo"),
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(3),
				OutputTokens: aws.Int32(2),
				TotalTokens:  aws.Int32(5),
			},
		}},
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}},
	)

	var chunks []string
	req := &ai.ModelRequest{}
	resp, err := (&Bedrock{}).consumeStreamEvents(context.Background(), events, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) != 1 || !chunk.Content[0].IsText() {
			t.Fatalf("chunk content = %+v, want one text part", chunk.Content)
		}
		chunks = append(chunks, chunk.Content[0].Text)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(chunks, "") != "hello" {
		t.Fatalf("streamed chunks = %q, want hello", strings.Join(chunks, ""))
	}
	if resp.Text() != "hello" {
		t.Fatalf("final text = %q, want hello", resp.Text())
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %+v, want total tokens 5", resp.Usage)
	}
	if resp.Request != req {
		t.Fatalf("Request = %p, want %p", resp.Request, req)
	}
}

func TestBlocksToParts_StreamReassembly(t *testing.T) {
	blocks := map[int32]*streamBlock{
		1: {isTool: true, toolID: "call_1", toolName: "get_weather"},
		0: {},
	}
	blocks[1].toolInput.WriteString(`{"location":"NYC"}`)
	blocks[0].text.WriteString("Looking up the weather...")

	parts, err := (&Bedrock{}).blocksToParts(blocks, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Text != "Looking up the weather..." {
		t.Fatalf("parts[0] = %q, want text", parts[0].Text)
	}
	if !parts[1].IsToolRequest() {
		t.Fatalf("parts[1] = %+v, want tool request", parts[1])
	}
	if parts[1].ToolRequest.Name != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", parts[1].ToolRequest.Name)
	}
	input := parts[1].ToolRequest.Input.(map[string]any)
	if input["location"] != "NYC" {
		t.Fatalf("location = %v, want NYC", input["location"])
	}
}

func TestConsumeStreamEvents_ToolUseFragmentsAndSchemaConversion(t *testing.T) {
	req := &ai.ModelRequest{Tools: []*ai.ToolDefinition{
		{
			Name: "get_weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
					"count":    map[string]any{"type": "string"},
				},
			},
		},
	}}
	events := streamEvents(
		toolStart(0, "call_1", "get_weather"),
		toolDelta(0, `{"location":"NYC",`),
		toolDelta(0, `"count":42}`),
		toolStop(0),
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
	)

	var toolChunks int
	resp, err := (&Bedrock{}).consumeStreamEvents(context.Background(), events, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		toolChunks++
		if len(chunk.Content) != 1 || !chunk.Content[0].IsToolRequest() {
			t.Fatalf("chunk content = %+v, want one tool request", chunk.Content)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if toolChunks != 1 {
		t.Fatalf("tool callback count = %d, want 1", toolChunks)
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if len(resp.Message.Content) != 1 || !resp.Message.Content[0].IsToolRequest() {
		t.Fatalf("final content = %+v, want one tool request", resp.Message.Content)
	}
	input := resp.Message.Content[0].ToolRequest.Input.(map[string]any)
	if input["count"] != "42" {
		t.Fatalf("count = %#v (%T), want string 42", input["count"], input["count"])
	}
}

func TestConsumeStreamEvents_CallbackErrors(t *testing.T) {
	callbackErr := errors.New("callback failed")
	tests := []struct {
		name   string
		events <-chan types.ConverseStreamOutput
	}{
		{
			name:   "text",
			events: streamEvents(textDelta(0, "hello")),
		},
		{
			name: "reasoning",
			events: streamEvents(&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberReasoningContent{
					Value: &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"},
				},
			}}),
		},
		{
			name: "tool stop",
			events: streamEvents(
				toolStart(0, "call_1", "get_weather"),
				toolDelta(0, `{}`),
				toolStop(0),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (&Bedrock{}).consumeStreamEvents(context.Background(), tt.events, nil, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				return callbackErr
			})
			if !errors.Is(err, callbackErr) {
				t.Fatalf("error = %v, want callback error", err)
			}
		})
	}
}

func TestConsumeStreamEvents_MalformedToolInputErrors(t *testing.T) {
	var callbacks int
	_, err := (&Bedrock{}).consumeStreamEvents(context.Background(), streamEvents(
		toolStart(0, "call_1", "get_weather"),
		toolDelta(0, `{not valid`),
		toolStop(0),
	), nil, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		callbacks++
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "stream tool block 0") {
		t.Fatalf("error = %v, want malformed tool input error", err)
	}
	if callbacks != 0 {
		t.Fatalf("callbacks = %d, want 0", callbacks)
	}
}

func TestConsumeStreamEvents_UnsupportedDeltaErrors(t *testing.T) {
	_, err := (&Bedrock{}).consumeStreamEvents(context.Background(), streamEvents(
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &types.ContentBlockDeltaMemberCitation{},
		}},
	), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "unhandled stream content delta") {
		t.Fatalf("error = %v, want unsupported delta error", err)
	}
}

func TestConsumeStreamEvents_EmptyContentReturnsPlaceholder(t *testing.T) {
	resp, err := (&Bedrock{}).consumeStreamEvents(context.Background(), streamEvents(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.Content) != 1 || !resp.Message.Content[0].IsText() || resp.Message.Content[0].Text != "" {
		t.Fatalf("content = %+v, want empty text placeholder", resp.Message.Content)
	}
}

func TestDecodeToolInput_EmptyAndMalformed(t *testing.T) {
	v, err := decodeToolInput(" \n\t")
	if err != nil || v != nil {
		t.Fatalf("decodeToolInput(empty) = (%v, %v), want (nil, nil)", v, err)
	}
	if _, err := decodeToolInput("{not valid"); err == nil {
		t.Fatal("expected malformed JSON error")
	}
}

func streamEvents(events ...types.ConverseStreamOutput) <-chan types.ConverseStreamOutput {
	ch := make(chan types.ConverseStreamOutput, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return ch
}

func textDelta(idx int32, text string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(idx),
		Delta:             &types.ContentBlockDeltaMemberText{Value: text},
	}}
}

func toolStart(idx int32, id, name string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: aws.Int32(idx),
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: aws.String(id),
			Name:      aws.String(name),
		}},
	}}
}

func toolDelta(idx int32, input string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(idx),
		Delta:             &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(input)}},
	}}
}

func toolStop(idx int32) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{
		ContentBlockIndex: aws.Int32(idx),
	}}
}
