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
	"encoding/base64"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
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
		// Claude 4/4.5/4.6 family
		{"anthropic.claude-haiku-4-5-20251001-v1:0", true, true},
		{"anthropic.claude-opus-4-1-20250805-v1:0", true, true},
		{"anthropic.claude-opus-4-20250514-v1:0", true, true},
		{"anthropic.claude-sonnet-4-20250514-v1:0", true, true},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", true, true},
		{"anthropic.claude-opus-4-5-20251101-v1:0", true, true},
		{"anthropic.claude-sonnet-4-6", true, true},
		{"anthropic.claude-opus-4-6-v1", true, true},
		// Provisioned-throughput variants
		{"anthropic.claude-3-haiku-20240307-v1:0:48k", true, true},
		{"anthropic.claude-3-sonnet-20240229-v1:0:200k", true, true},
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

func TestBuildConverseInput_MediaContentBlocks(t *testing.T) {
	b := &Bedrock{initted: true}

	pdfData := []byte("%PDF-1.4 test content")
	pdfB64 := base64.StdEncoding.EncodeToString(pdfData)

	imgData := []byte("\x89PNG test image")
	imgB64 := base64.StdEncoding.EncodeToString(imgData)

	txtData := []byte("plain text content")
	txtB64 := base64.StdEncoding.EncodeToString(txtData)

	tests := []struct {
		name            string
		contentType     string
		dataURL         string
		wantBlockType   string // "document" or "image"
		wantDocFormat   types.DocumentFormat
		wantImageFormat types.ImageFormat
	}{
		{
			name:          "PDF produces DocumentBlock with pdf format",
			contentType:   "application/pdf",
			dataURL:       "data:application/pdf;base64," + pdfB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatPdf,
		},
		{
			name:          "text/plain produces DocumentBlock with txt format",
			contentType:   "text/plain",
			dataURL:       "data:text/plain;base64," + txtB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatTxt,
		},
		{
			name:          "text/markdown produces DocumentBlock with md format",
			contentType:   "text/markdown",
			dataURL:       "data:text/markdown;base64," + txtB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatMd,
		},
		{
			name:            "image/png produces ImageBlock with png format",
			contentType:     "image/png",
			dataURL:         "data:image/png;base64," + imgB64,
			wantBlockType:   "image",
			wantImageFormat: types.ImageFormatPng,
		},
		{
			name:            "image/jpeg produces ImageBlock with jpeg format",
			contentType:     "image/jpeg",
			dataURL:         "data:image/jpeg;base64," + imgB64,
			wantBlockType:   "image",
			wantImageFormat: types.ImageFormatJpeg,
		},
		{
			name:            "unknown type falls back to ImageBlock with png format",
			contentType:     "application/octet-stream",
			dataURL:         "data:application/octet-stream;base64," + imgB64,
			wantBlockType:   "image",
			wantImageFormat: types.ImageFormatPng,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewMediaPart(tt.contentType, tt.dataURL),
						},
					},
				},
			}

			converseInput, err := b.buildConverseInput("anthropic.claude-3-5-sonnet-20241022-v2:0", req)
			if err != nil {
				t.Fatalf("buildConverseInput() error = %v", err)
			}

			if len(converseInput.Messages) == 0 || len(converseInput.Messages[0].Content) == 0 {
				t.Fatal("expected at least one content block in the message")
			}

			block := converseInput.Messages[0].Content[0]

			switch tt.wantBlockType {
			case "document":
				docBlock, ok := block.(*types.ContentBlockMemberDocument)
				if !ok {
					t.Fatalf("expected *types.ContentBlockMemberDocument, got %T", block)
				}
				if docBlock.Value.Format != tt.wantDocFormat {
					t.Errorf("document format = %v, want %v", docBlock.Value.Format, tt.wantDocFormat)
				}
			case "image":
				imgBlock, ok := block.(*types.ContentBlockMemberImage)
				if !ok {
					t.Fatalf("expected *types.ContentBlockMemberImage, got %T", block)
				}
				if imgBlock.Value.Format != tt.wantImageFormat {
					t.Errorf("image format = %v, want %v", imgBlock.Value.Format, tt.wantImageFormat)
				}
			}
		})
	}
}
