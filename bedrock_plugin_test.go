// Copyright 2025 Xavier Portilla Edo
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
	"testing"
)

func TestInferModelCapabilities_WithInferenceProfiles(t *testing.T) {
	b := &Bedrock{}

	tests := []struct {
		name             string
		modelName        string
		modelType        string
		expectTools      bool
		expectMultimodal bool
	}{
		// Direct model IDs (no prefix)
		{
			name:             "claude-3-haiku direct - has both capabilities",
			modelName:        "anthropic.claude-3-haiku-20240307-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		{
			name:             "claude-3-5-haiku direct - tools only",
			modelName:        "anthropic.claude-3-5-haiku-20241022-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: false,
		},
		{
			name:             "nova-micro direct - tools only",
			modelName:        "amazon.nova-micro-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: false,
		},
		// With inference profile prefixes
		{
			name:             "us prefix - claude-3-haiku",
			modelName:        "us.anthropic.claude-3-haiku-20240307-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		{
			name:             "eu prefix - claude-3-5-sonnet",
			modelName:        "eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		{
			name:             "global prefix - nova-pro",
			modelName:        "global.amazon.nova-pro-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		{
			name:             "apac prefix - llama3-2-11b (multimodal llama)",
			modelName:        "apac.meta.llama3-2-11b-instruct-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		{
			name:             "jp prefix - llama3-8b (tools only)",
			modelName:        "jp.meta.llama3-8b-instruct-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: false,
		},
		{
			name:             "us-gov prefix - claude-opus-4",
			modelName:        "us-gov.anthropic.claude-opus-4-20250514-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectMultimodal: true,
		},
		// Unknown models (not in capability map)
		{
			name:             "unknown model - no capabilities",
			modelName:        "unknown.model-v1:0",
			modelType:        "chat",
			expectTools:      false,
			expectMultimodal: false,
		},
		{
			name:             "unknown model with prefix - no capabilities",
			modelName:        "us.unknown.model-v1:0",
			modelType:        "chat",
			expectTools:      false,
			expectMultimodal: false,
		},
		// Image and embedding types should ignore capability map
		{
			name:             "image model type - always has media output",
			modelName:        "amazon.titan-image-generator-v1",
			modelType:        "image",
			expectTools:      false,
			expectMultimodal: true, // Media output capability
		},
		{
			name:             "embedding model type - no capabilities",
			modelName:        "amazon.titan-embed-text-v1",
			modelType:        "embedding",
			expectTools:      false,
			expectMultimodal: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := b.inferModelCapabilities(tt.modelName, tt.modelType)

			if info.Supports.Tools != tt.expectTools {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.Tools = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.Tools, tt.expectTools)
			}

			if info.Supports.Media != tt.expectMultimodal {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.Media = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.Media, tt.expectMultimodal)
			}

			// Verify the label preserves the original model name (including prefix)
			if info.Label != tt.modelName {
				t.Errorf("inferModelCapabilities(%q, %q).Label = %q, want %q",
					tt.modelName, tt.modelType, info.Label, tt.modelName)
			}
		})
	}
}

func TestInferenceProfilePrefixes_Coverage(t *testing.T) {
	// Verify all documented prefixes are in the list
	expectedPrefixes := []string{
		"global.",
		"us.",
		"eu.",
		"jp.",
		"apac.",
		"au.",
		"us-gov.",
	}

	if len(inferenceProfilePrefixes) != len(expectedPrefixes) {
		t.Errorf("inferenceProfilePrefixes has %d entries, expected %d",
			len(inferenceProfilePrefixes), len(expectedPrefixes))
	}

	for _, expected := range expectedPrefixes {
		found := false
		for _, prefix := range inferenceProfilePrefixes {
			if prefix == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected prefix %q not found in inferenceProfilePrefixes", expected)
		}
	}
}

func TestModelCapabilities_KnownModels(t *testing.T) {
	// Verify some known models are in the capability map with correct values
	tests := []struct {
		modelID          string
		expectMultimodal bool
		expectTools      bool
	}{
		// Claude 3 family
		{"anthropic.claude-3-haiku-20240307-v1:0", true, true},
		{"anthropic.claude-3-5-haiku-20241022-v1:0", false, true}, // 3.5 Haiku is text-only
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", true, true},
		{"anthropic.claude-3-7-sonnet-20250219-v1:0", true, true},
		// Claude 4 family
		{"anthropic.claude-opus-4-20250514-v1:0", true, true},
		{"anthropic.claude-sonnet-4-20250514-v1:0", true, true},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", true, true},
		{"anthropic.claude-opus-4-5-20251101-v1:0", true, true},
		// Nova family
		{"amazon.nova-micro-v1:0", false, true}, // Micro is text-only
		{"amazon.nova-lite-v1:0", true, true},
		{"amazon.nova-pro-v1:0", true, true},
		// Llama family - mixed multimodal support
		{"meta.llama3-8b-instruct-v1:0", false, true},
		{"meta.llama3-2-11b-instruct-v1:0", true, true}, // 11b and 90b are multimodal
		{"meta.llama3-2-90b-instruct-v1:0", true, true},
		{"meta.llama4-maverick-17b-instruct-v1:0", true, true},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			caps, found := modelCapabilities[tt.modelID]
			if !found {
				t.Fatalf("model %q not found in modelCapabilities", tt.modelID)
			}

			if caps.Multimodal != tt.expectMultimodal {
				t.Errorf("modelCapabilities[%q].Multimodal = %v, want %v",
					tt.modelID, caps.Multimodal, tt.expectMultimodal)
			}

			if caps.Tools != tt.expectTools {
				t.Errorf("modelCapabilities[%q].Tools = %v, want %v",
					tt.modelID, caps.Tools, tt.expectTools)
			}
		})
	}
}
