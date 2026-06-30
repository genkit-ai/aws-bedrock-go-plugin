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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
)

var errStreamBlockRequired = errors.New("bedrock: stream block is nil")

func (b *Bedrock) generateTextStream(ctx context.Context, input *bedrockruntime.ConverseInput, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	streamInput := &bedrockruntime.ConverseStreamInput{
		ModelId:                      input.ModelId,
		Messages:                     input.Messages,
		System:                       input.System,
		InferenceConfig:              input.InferenceConfig,
		ToolConfig:                   input.ToolConfig,
		AdditionalModelRequestFields: input.AdditionalModelRequestFields,
	}

	streamOutput, err := b.client.ConverseStream(ctx, streamInput)
	if err != nil {
		return nil, fmt.Errorf("bedrock converse stream failed: %w", err)
	}
	stream := streamOutput.GetStream()
	if stream == nil {
		return nil, fmt.Errorf("bedrock converse stream is nil")
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			_ = closeErr
		}
	}()

	finalResponse, err := b.consumeStreamEvents(ctx, stream.Events(), originalInput, cb)
	if err != nil {
		return nil, err
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("bedrock converse stream error: %w", err)
	}
	return finalResponse, nil
}

// streamBlock accumulates the state of a single content block across delta
// events keyed by ContentBlockIndex.
type streamBlock struct {
	text               strings.Builder
	reasoning          strings.Builder
	reasoningSignature string
	redactedReasoning  []byte
	toolID             string
	toolName           string
	toolInput          strings.Builder
	isTool             bool
}

func (b *Bedrock) consumeStreamEvents(ctx context.Context, events <-chan types.ConverseStreamOutput, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	blocks := map[int32]*streamBlock{}
	var stopReason types.StopReason
	var usage *types.TokenUsage

	for event := range events {
		switch e := event.(type) {
		case *types.ConverseStreamOutputMemberMessageStart:
			// The outbound role is always assistant/model.
		case *types.ConverseStreamOutputMemberContentBlockStart:
			idx := indexOf(e.Value.ContentBlockIndex)
			block := getOrInit(blocks, idx)
			if startTool, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
				block.isTool = true
				block.toolID = aws.ToString(startTool.Value.ToolUseId)
				block.toolName = aws.ToString(startTool.Value.Name)
			}
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if e.Value.Delta == nil {
				continue
			}
			idx := indexOf(e.Value.ContentBlockIndex)
			block := getOrInit(blocks, idx)
			if err := appendContentBlockDelta(ctx, block, e.Value.Delta, cb); err != nil {
				return nil, err
			}
		case *types.ConverseStreamOutputMemberContentBlockStop:
			idx := indexOf(e.Value.ContentBlockIndex)
			if err := b.emitToolBlockStop(ctx, idx, blocks[idx], originalInput, cb); err != nil {
				return nil, err
			}
		case *types.ConverseStreamOutputMemberMessageStop:
			stopReason = e.Value.StopReason
		case *types.ConverseStreamOutputMemberMetadata:
			usage = e.Value.Usage
		default:
			// Unknown top-level events are ignored so new Bedrock event types don't break streaming.
		}
	}

	parts, err := b.blocksToParts(blocks, originalInput)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		parts = append(parts, ai.NewTextPart(""))
	}
	finishReason := convertStopReasonToGenkit(stopReason)
	if stopReason == "" {
		finishReason = ai.FinishReasonStop
	}
	return &ai.ModelResponse{
		Message:      &ai.Message{Role: ai.RoleModel, Content: parts},
		FinishReason: finishReason,
		Usage:        usageFromTokens(usage),
		Request:      originalInput,
	}, nil
}

// blocksToParts assembles accumulated stream state in ContentBlockIndex order.
func (b *Bedrock) blocksToParts(blocks map[int32]*streamBlock, originalInput *ai.ModelRequest) ([]*ai.Part, error) {
	idxs := make([]int32, 0, len(blocks))
	for idx := range blocks {
		idxs = append(idxs, idx)
	}
	sort.Slice(idxs, func(i, j int) bool { return idxs[i] < idxs[j] })

	parts := make([]*ai.Part, 0, len(idxs))
	for _, idx := range idxs {
		block := blocks[idx]
		if block == nil {
			continue
		}
		if block.isTool {
			part, err := b.toolBlockToPart(idx, block, originalInput)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
			continue
		}
		if block.reasoning.Len() > 0 || len(block.redactedReasoning) > 0 {
			parts = append(parts, newBedrockReasoningPart(block.reasoning.String(), block.reasoningSignature, block.redactedReasoning))
		}
		if block.text.Len() > 0 {
			parts = append(parts, ai.NewTextPart(block.text.String()))
		}
	}
	return parts, nil
}

func appendContentBlockDelta(ctx context.Context, block *streamBlock, delta types.ContentBlockDelta, cb func(context.Context, *ai.ModelResponseChunk) error) error {
	if block == nil {
		return errStreamBlockRequired
	}
	switch d := delta.(type) {
	case *types.ContentBlockDeltaMemberText:
		block.text.WriteString(d.Value)
		if cb != nil {
			if err := cb(ctx, &ai.ModelResponseChunk{Index: 0, Content: []*ai.Part{ai.NewTextPart(d.Value)}}); err != nil {
				return fmt.Errorf("callback error: %w", err)
			}
		}
	case *types.ContentBlockDeltaMemberToolUse:
		block.isTool = true
		block.toolInput.WriteString(aws.ToString(d.Value.Input))
	case *types.ContentBlockDeltaMemberReasoningContent:
		part, err := appendReasoningDelta(block, d.Value)
		if err != nil {
			return err
		}
		if part != nil && cb != nil {
			if err := cb(ctx, &ai.ModelResponseChunk{Index: 0, Content: []*ai.Part{part}}); err != nil {
				return fmt.Errorf("callback error: %w", err)
			}
		}
	default:
		return fmt.Errorf("bedrock: unhandled stream content delta variant %T", delta)
	}
	return nil
}

func appendReasoningDelta(block *streamBlock, delta types.ReasoningContentBlockDelta) (*ai.Part, error) {
	if block == nil {
		return nil, errStreamBlockRequired
	}
	switch d := delta.(type) {
	case *types.ReasoningContentBlockDeltaMemberText:
		block.reasoning.WriteString(d.Value)
		return newBedrockReasoningPart(d.Value, "", nil), nil
	case *types.ReasoningContentBlockDeltaMemberSignature:
		block.reasoningSignature = d.Value
	case *types.ReasoningContentBlockDeltaMemberRedactedContent:
		block.redactedReasoning = append(block.redactedReasoning, d.Value...)
	default:
		return nil, fmt.Errorf("bedrock: unhandled stream reasoning delta variant %T", delta)
	}
	return nil, nil
}

func (b *Bedrock) emitToolBlockStop(ctx context.Context, idx int32, block *streamBlock, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) error {
	if block == nil || !block.isTool || cb == nil {
		return nil
	}
	part, err := b.toolBlockToPart(idx, block, originalInput)
	if err != nil {
		return err
	}
	if err := cb(ctx, &ai.ModelResponseChunk{Index: 0, Content: []*ai.Part{part}}); err != nil {
		return fmt.Errorf("callback error: %w", err)
	}
	return nil
}

func (b *Bedrock) toolBlockToPart(idx int32, block *streamBlock, originalInput *ai.ModelRequest) (*ai.Part, error) {
	input, err := decodeToolInput(block.toolInput.String())
	if err != nil {
		return nil, fmt.Errorf("bedrock: stream tool block %d: %w", idx, err)
	}
	if inputMap, ok := input.(map[string]any); ok {
		var tools []*ai.ToolDefinition
		if originalInput != nil {
			tools = originalInput.Tools
		}
		input = b.convertToolInputTypes(inputMap, block.toolName, tools)
	}
	return ai.NewToolRequestPart(&ai.ToolRequest{
		Ref:   block.toolID,
		Name:  block.toolName,
		Input: input,
	}), nil
}

func decodeToolInput(s string) (any, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	decoder := json.NewDecoder(strings.NewReader(s))
	decoder.UseNumber()
	var v any
	if err := decoder.Decode(&v); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&v); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("bedrock: tool input contains trailing data after JSON value")
		}
		return nil, err
	}
	return v, nil
}

func getOrInit(blocks map[int32]*streamBlock, idx int32) *streamBlock {
	if block, ok := blocks[idx]; ok {
		return block
	}
	block := &streamBlock{}
	blocks[idx] = block
	return block
}

func indexOf(idx *int32) int32 {
	if idx == nil {
		return 0
	}
	return *idx
}
