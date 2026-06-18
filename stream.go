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

	// Build final response
	var fullText strings.Builder
	var finalResponse *ai.ModelResponse
	var stopReason types.StopReason

	// Process stream events
	for event := range stream.Events() {
		switch e := event.(type) {

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			// Text delta received
			deltaEvent := e.Value
			if deltaEvent.Delta != nil {
				if textDelta, ok := deltaEvent.Delta.(*types.ContentBlockDeltaMemberText); ok {
					text := textDelta.Value
					fullText.WriteString(text)

					// Send chunk to callback
					chunk := &ai.ModelResponseChunk{
						Index: 0,
						Content: []*ai.Part{
							ai.NewTextPart(text),
						},
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
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewTextPart(fullText.String()),
					},
				},
				FinishReason: convertStopReasonToGenkit(stopReason),
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
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewTextPart(fullText.String()),
				},
			},
			FinishReason: ai.FinishReasonStop,
		}
	}

	return finalResponse, nil
}
