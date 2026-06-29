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
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
)

func (b *Bedrock) generateTextStream(ctx context.Context, input *bedrockruntime.ConverseInput, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	ctx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	// Convert ConverseInput to ConverseStreamInput
	streamInput := &bedrockruntime.ConverseStreamInput{
		ModelId:                      input.ModelId,
		Messages:                     input.Messages,
		System:                       input.System,
		InferenceConfig:              input.InferenceConfig,
		ToolConfig:                   input.ToolConfig,
		AdditionalModelRequestFields: input.AdditionalModelRequestFields,
	}

	// Call Bedrock ConverseStream API
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
			// Log the error but don't fail the operation
			// In a real implementation, you might want to use a proper logger
			_ = closeErr
		}
	}()

	// Accumulate streamed content. Reasoning ("thinking") deltas are tracked
	// separately from plain text so the signed/redacted reasoning can be
	// attached to the final response and replayed on the next turn.
	acc := &streamAccumulator{}
	var finalResponse *ai.ModelResponse
	var stopReason types.StopReason

	// Process stream events
	for event := range stream.Events() {
		switch e := event.(type) {

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			deltaEvent := e.Value
			if deltaEvent.Delta == nil {
				continue
			}
			switch delta := deltaEvent.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				acc.text.WriteString(delta.Value)
				chunk := &ai.ModelResponseChunk{
					Index:   0,
					Content: []*ai.Part{ai.NewTextPart(delta.Value)},
				}
				if err := cb(ctx, chunk); err != nil {
					return nil, fmt.Errorf("callback error: %w", err)
				}

			case *types.ContentBlockDeltaMemberReasoningContent:
				part, err := appendReasoningDelta(acc, delta.Value)
				if err != nil {
					return nil, err
				}
				if part != nil {
					chunk := &ai.ModelResponseChunk{
						Index:   0,
						Content: []*ai.Part{part},
					}
					if err := cb(ctx, chunk); err != nil {
						return nil, fmt.Errorf("callback error: %w", err)
					}
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			// Message ended - prepare final response
			stopEvent := e.Value
			stopReason = stopEvent.StopReason

			finalResponse = &ai.ModelResponse{
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: acc.finalContent(),
				},
				FinishReason: convertStopReasonToGenkit(stopReason),
				Request:      originalInput,
			}

		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("bedrock converse stream error: %w", err)
	}

	// Return final response
	if finalResponse == nil {
		finalResponse = &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: acc.finalContent(),
			},
			FinishReason: ai.FinishReasonStop,
			Request:      originalInput,
		}
	}

	return finalResponse, nil
}

// streamAccumulator collects streamed content across delta events. The current
// streaming path reconstructs a single text block plus any reasoning;
// block-indexed reassembly (e.g. for streamed tool-use) is a separate concern.
type streamAccumulator struct {
	text               strings.Builder
	reasoning          strings.Builder
	reasoningSignature string
	redactedReasoning  []byte
}

// appendReasoningDelta folds a reasoning delta into the accumulator. Text deltas
// are accumulated and returned as an emittable chunk; signature and redacted
// deltas are accumulated silently (nil part) since they only matter for the
// final, replayable reasoning part.
func appendReasoningDelta(acc *streamAccumulator, delta types.ReasoningContentBlockDelta) (*ai.Part, error) {
	switch d := delta.(type) {
	case *types.ReasoningContentBlockDeltaMemberText:
		acc.reasoning.WriteString(d.Value)
		return newBedrockReasoningPart(d.Value, "", nil), nil
	case *types.ReasoningContentBlockDeltaMemberSignature:
		acc.reasoningSignature = d.Value
		return nil, nil
	case *types.ReasoningContentBlockDeltaMemberRedactedContent:
		acc.redactedReasoning = append(acc.redactedReasoning, d.Value...)
		return nil, nil
	default:
		return nil, fmt.Errorf("bedrock: unhandled stream reasoning delta variant %T", delta)
	}
}

// finalContent assembles the accumulated stream state into response parts. Any
// reasoning precedes the text so the assistant turn replays in the order
// thinking models require.
func (acc *streamAccumulator) finalContent() []*ai.Part {
	var parts []*ai.Part
	if acc.reasoning.Len() > 0 || len(acc.redactedReasoning) > 0 {
		parts = append(parts, newBedrockReasoningPart(acc.reasoning.String(), acc.reasoningSignature, acc.redactedReasoning))
	}
	parts = append(parts, ai.NewTextPart(acc.text.String()))
	return parts
}
