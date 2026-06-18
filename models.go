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
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// inferenceProfilePrefixes lists all valid inference profile region prefixes.
// When an inference profile is passed instead of a model ID, the prefix is
// stripped before checking capabilities.
var inferenceProfilePrefixes = []string{
	"global.",
	"us.",
	"eu.",
	"jp.",
	"apac.",
	"au.",
	"us-gov.",
}

// modelCapabilities maps model IDs to their capabilities.
// This consolidates the previous multimodalModels and toolSupportedModels lists.
var modelCapabilities = map[string]ModelCapability{
	// Anthropic Claude 3 models
	"anthropic.claude-3-haiku-20240307-v1:0":    {Multimodal: true, Tools: true},
	"anthropic.claude-3-sonnet-20240229-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-3-opus-20240229-v1:0":     {Multimodal: true, Tools: true},
	"anthropic.claude-3-5-haiku-20241022-v1:0":  {Multimodal: false, Tools: true},
	"anthropic.claude-3-5-sonnet-20240620-v1:0": {Multimodal: true, Tools: true},
	"anthropic.claude-3-5-sonnet-20241022-v2:0": {Multimodal: true, Tools: true},
	"anthropic.claude-3-7-sonnet-20250219-v1:0": {Multimodal: true, Tools: true},
	// Anthropic Claude 4/4.5/4.6 models
	"anthropic.claude-haiku-4-5-20251001-v1:0":  {Multimodal: true, Tools: true},
	"anthropic.claude-opus-4-1-20250805-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-opus-4-20250514-v1:0":     {Multimodal: true, Tools: true},
	"anthropic.claude-sonnet-4-20250514-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-sonnet-4-5-20250929-v1:0": {Multimodal: true, Tools: true},
	"anthropic.claude-opus-4-5-20251101-v1:0":   {Multimodal: true, Tools: true},
	"anthropic.claude-sonnet-4-6":               {Multimodal: true, Tools: true},
	"anthropic.claude-opus-4-6-v1":              {Multimodal: true, Tools: true},
	// Provisioned-throughput variants (28k/48k/200k context)
	"anthropic.claude-3-haiku-20240307-v1:0:48k":   {Multimodal: true, Tools: true},
	"anthropic.claude-3-haiku-20240307-v1:0:200k":  {Multimodal: true, Tools: true},
	"anthropic.claude-3-sonnet-20240229-v1:0:28k":  {Multimodal: true, Tools: true},
	"anthropic.claude-3-sonnet-20240229-v1:0:200k": {Multimodal: true, Tools: true},
	// Amazon Nova models
	"amazon.nova-micro-v1:0":   {Multimodal: false, Tools: true},
	"amazon.nova-lite-v1:0":    {Multimodal: true, Tools: true},
	"amazon.nova-pro-v1:0":     {Multimodal: true, Tools: true},
	"amazon.nova-premier-v1:0": {Multimodal: true, Tools: true},
	// Cohere Command models
	"cohere.command-r-v1:0":      {Multimodal: false, Tools: true},
	"cohere.command-r-plus-v1:0": {Multimodal: false, Tools: true},
	// Mistral models
	"mistral.mistral-large-2402-v1:0": {Multimodal: false, Tools: true},
	"mistral.mistral-large-2407-v1:0": {Multimodal: false, Tools: true},
	"mistral.mistral-small-2402-v1:0": {Multimodal: false, Tools: true},
	"mistral.pixtral-large-2502-v1:0": {Multimodal: true, Tools: true},
	// AI21 Labs Jamba models
	"ai21.jamba-1-5-large-v1:0": {Multimodal: false, Tools: true},
	"ai21.jamba-1-5-mini-v1:0":  {Multimodal: false, Tools: true},
	// Meta Llama models
	"meta.llama3-8b-instruct-v1:0":           {Multimodal: false, Tools: true},
	"meta.llama3-70b-instruct-v1:0":          {Multimodal: false, Tools: true},
	"meta.llama3-1-8b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-1-70b-instruct-v1:0":        {Multimodal: false, Tools: true},
	"meta.llama3-1-405b-instruct-v1:0":       {Multimodal: false, Tools: true},
	"meta.llama3-2-1b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-2-3b-instruct-v1:0":         {Multimodal: false, Tools: true},
	"meta.llama3-2-11b-instruct-v1:0":        {Multimodal: true, Tools: true},
	"meta.llama3-2-90b-instruct-v1:0":        {Multimodal: true, Tools: true},
	"meta.llama3-3-70b-instruct-v1:0":        {Multimodal: false, Tools: true},
	"meta.llama4-maverick-17b-instruct-v1:0": {Multimodal: true, Tools: true},
	"meta.llama4-scout-17b-instruct-v1:0":    {Multimodal: true, Tools: true},
	// DeepSeek models
	"deepseek.r1-v1:0": {Multimodal: false, Tools: true},
	// Writer models
	"writer.palmyra-x4-v1:0": {Multimodal: false, Tools: true},
	"writer.palmyra-x5-v1:0": {Multimodal: false, Tools: true},
	// TwelveLabs models
	"twelvelabs.pegasus-1-2-v1:0": {Multimodal: false, Tools: true},
}

// inferModelCapabilities infers model capabilities based on model name and type.
// It strips any inference profile prefix before looking up capabilities.
func (b *Bedrock) inferModelCapabilities(modelName, modelType string) *ai.ModelInfo {
	// Strip inference profile prefix to get base model ID for capability lookup
	baseModelID := b.stripInferenceProfilePrefix(modelName)

	// Look up capabilities from the map
	caps, found := modelCapabilities[baseModelID]

	switch modelType {
	case "image":
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  false,
				Tools:      false,
				SystemRole: false,
				Media:      true, // Can output images
			},
		}
	case "embedding":
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  false,
				Tools:      false,
				SystemRole: false,
				Media:      false,
			},
		}
	default: // chat, text models
		return &ai.ModelInfo{
			Label: modelName,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      found && caps.Tools,
				SystemRole: true,
				Media:      found && caps.Multimodal,
			},
		}
	}
}

func (b *Bedrock) stripInferenceProfilePrefix(modelID string) string {
	for _, prefix := range inferenceProfilePrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return strings.TrimPrefix(modelID, prefix)
		}
	}
	return modelID
}
