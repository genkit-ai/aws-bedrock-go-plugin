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
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
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

// ModelDefinition represents a model with its name and type.
type ModelDefinition struct {
	Name string // Model ID as used in AWS Bedrock
	Type string // Type: "chat", "text", "image", "embedding"
}
