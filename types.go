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

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// Type aliases for better readability
type (
	BedrockClient = *bedrockruntime.Client
	Role          = ai.Role
	ToolChoice    = string
	FinishReason  = string
)

// ModelCapability represents the capabilities of a model
type ModelCapability struct {
	Multimodal bool // Supports image/media inputs
	Tools      bool // Supports function calling
}

// Constants
const provider = "bedrock"

// Role constants (would come from ai package)
const (
	RoleUser   Role = "user"
	RoleModel  Role = "assistant"
	RoleSystem Role = "system"
	RoleTool   Role = "tool"
)

// Tool choice constants
const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceRequired ToolChoice = "required"
	ToolChoiceNone     ToolChoice = "none"
)

// Finish reason constants
const (
	FinishReasonStop    FinishReason = "stop"
	FinishReasonLength  FinishReason = "length"
	FinishReasonBlocked FinishReason = "blocked"
	FinishReasonOther   FinishReason = "other"
	FinishReasonUnknown FinishReason = "unknown"
)

const bedrockCachePointTypeKey = "bedrockCachePointType"

// Metadata keys used to round-trip Bedrock reasoning ("thinking") content back
// into a follow-up request. Bedrock returns signed and sometimes redacted
// reasoning that must be replayed verbatim on the next turn or the model
// rejects it, so the signature and redacted bytes are stashed on the
// ai.Part metadata. These keys are Bedrock-specific: a generic reasoning part
// created via ai.NewReasoningPart (without these) is intentionally NOT
// round-tripped, so foreign reasoning can't corrupt a Bedrock conversation.
const (
	reasoningSignatureMetadataKey = "bedrockReasoningSignature"
	redactedReasoningMetadataKey  = "bedrockRedactedContent"
)

// Config is the per-call configuration for Bedrock Converse models. Pass it
// via [ai.WithConfig].
//
// It is fully optional and additive: callers may still pass configuration as a
// map[string]any (the historical shape) or as *ai.GenerationCommonConfig; see
// configFromRequest. The typed form exists mainly so model-specific knobs like
// Claude extended thinking can be enabled through AdditionalModelRequestFields.
type Config struct {
	// MaxTokens is the upper bound on the generated response length. When 0 the
	// plugin leaves it unset and Bedrock applies its own per-model default.
	MaxTokens int `json:"maxTokens,omitempty"`

	// Temperature controls sampling randomness. nil leaves it to the model default.
	Temperature *float32 `json:"temperature,omitempty"`

	// TopP is the nucleus-sampling cutoff. nil leaves it to the model default.
	TopP *float32 `json:"topP,omitempty"`

	// StopSequences are strings that, when generated, halt generation.
	StopSequences []string `json:"stopSequences,omitempty"`

	// ToolChoice selects how the model should pick tools. It is accepted here
	// for forward compatibility; wiring it through the Converse request is a
	// separate change and it is currently not applied.
	ToolChoice string `json:"toolChoice,omitempty"`

	// AdditionalModelRequestFields is forwarded verbatim as the Converse API's
	// AdditionalModelRequestFields document. Use it for model-specific knobs not
	// covered by the inference-config surface, e.g. Claude extended thinking:
	//
	//	&bedrock.Config{
	//		MaxTokens: 8000,
	//		AdditionalModelRequestFields: map[string]any{
	//			"thinking": map[string]any{"type": "enabled", "budget_tokens": 5000},
	//		},
	//	}
	AdditionalModelRequestFields map[string]any `json:"additionalModelRequestFields,omitempty"`
}

// configSchema returns the JSON schema for [Config], used as the per-call
// ConfigSchema on every defined Converse model.
func configSchema() map[string]any { return core.InferSchemaMap(Config{}) }

// newBedrockReasoningPart builds an ai reasoning part carrying the Bedrock
// signature and/or redacted bytes needed to replay it on the next turn. The
// signature is also stored under the generic "signature" key (via
// ai.NewReasoningPart) so framework-level consumers see it too.
func newBedrockReasoningPart(text, signature string, redacted []byte) *ai.Part {
	var sig []byte
	if signature != "" {
		sig = []byte(signature)
	}
	p := ai.NewReasoningPart(text, sig)
	if len(sig) > 0 {
		p.Metadata[reasoningSignatureMetadataKey] = sig
	}
	if len(redacted) > 0 {
		p.Metadata[redactedReasoningMetadataKey] = redacted
	}
	return p
}

// metadataBytes reads a []byte value from part metadata, also accepting a
// base64-encoded string (which is how []byte survives a JSON round-trip on
// resumed/serialized flows).
func metadataBytes(metadata map[string]any, key string) []byte {
	if metadata == nil {
		return nil
	}
	switch v := metadata[key].(type) {
	case []byte:
		return v
	case string:
		b, err := base64.StdEncoding.DecodeString(v)
		if err != nil {
			return nil
		}
		return b
	default:
		return nil
	}
}

// ModelDefinition represents a model with its name and type.
type ModelDefinition struct {
	Name string // Model ID as used in AWS Bedrock
	Type string // Type: "chat", "text", "image", "embedding"
}
