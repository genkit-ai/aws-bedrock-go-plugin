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
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestInferModelCapabilities_WithInferenceProfiles(t *testing.T) {
	b := &Bedrock{}

	tests := []struct {
		name             string
		modelName        string
		modelType        string
		expectTools      bool
		expectToolChoice bool
		expectMultimodal bool
		expectStage      ai.ModelStage
	}{
		// Direct model IDs (no prefix)
		{
			name:             "claude-3-haiku direct - has both capabilities",
			modelName:        "anthropic.claude-3-haiku-20240307-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "claude-3-5-haiku direct - tools only",
			modelName:        "anthropic.claude-3-5-haiku-20241022-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: false,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "nova-micro direct - tools only",
			modelName:        "amazon.nova-micro-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: false,
			expectStage:      ai.ModelStageStable,
		},
		// With inference profile prefixes
		{
			name:             "us prefix - claude-3-haiku",
			modelName:        "us.anthropic.claude-3-haiku-20240307-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "eu prefix - claude-3-5-sonnet",
			modelName:        "eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "global prefix - nova-pro",
			modelName:        "global.amazon.nova-pro-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "apac prefix - llama3-2-11b (multimodal llama)",
			modelName:        "apac.meta.llama3-2-11b-instruct-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "jp prefix - llama3-8b (tools only)",
			modelName:        "jp.meta.llama3-8b-instruct-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: false,
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "us-gov prefix - claude-opus-4",
			modelName:        "us-gov.anthropic.claude-opus-4-20250514-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageStable,
		},
		// Unknown models (not in capability map)
		{
			name:             "unknown model - modern Converse defaults",
			modelName:        "unknown.model-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageUnstable,
		},
		{
			name:             "unknown model with prefix - modern Converse defaults",
			modelName:        "us.unknown.model-v1:0",
			modelType:        "chat",
			expectTools:      true,
			expectToolChoice: true,
			expectMultimodal: true,
			expectStage:      ai.ModelStageUnstable,
		},
		// Image and embedding types should ignore capability map
		{
			name:             "image model type - always has media output",
			modelName:        "amazon.titan-image-generator-v1",
			modelType:        "image",
			expectTools:      false,
			expectToolChoice: false,
			expectMultimodal: true, // Media output capability
			expectStage:      ai.ModelStageStable,
		},
		{
			name:             "embedding model type - no capabilities",
			modelName:        "amazon.titan-embed-text-v1",
			modelType:        "embedding",
			expectTools:      false,
			expectToolChoice: false,
			expectMultimodal: false,
			expectStage:      ai.ModelStageStable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := b.inferModelCapabilities(tt.modelName, tt.modelType)

			if info.Supports.Tools != tt.expectTools {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.Tools = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.Tools, tt.expectTools)
			}

			if info.Supports.ToolChoice != tt.expectToolChoice {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.ToolChoice = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.ToolChoice, tt.expectToolChoice)
			}

			if info.Supports.Media != tt.expectMultimodal {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.Media = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.Media, tt.expectMultimodal)
			}

			if info.Supports.Constrained != ai.ConstrainedSupportNone {
				t.Errorf("inferModelCapabilities(%q, %q).Supports.Constrained = %v, want %v",
					tt.modelName, tt.modelType, info.Supports.Constrained, ai.ConstrainedSupportNone)
			}

			if info.Stage != tt.expectStage {
				t.Errorf("inferModelCapabilities(%q, %q).Stage = %v, want %v",
					tt.modelName, tt.modelType, info.Stage, tt.expectStage)
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
		"us-gov.",
		"us.",
		"eu.",
		"jp.",
		"apac.",
		"au.",
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

func TestDefineModelRequiresInitializedPluginInstance(t *testing.T) {
	ctx := context.Background()
	b := &Bedrock{
		Region: "us-east-1",
		AWSConfig: &aws.Config{
			Region: "us-east-1",
			Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(
				"test-access-key",
				"test-secret-key",
				"",
			)),
		},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	if got := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-3-haiku-20240307-v1:0",
		Type: "chat",
	}, nil); got == nil {
		t.Fatal("DefineModel returned nil for initialized plugin")
	}

	assertPanicsWith(t, "bedrock: Init not called", func() {
		(&Bedrock{Region: "us-east-1"}).DefineModel(g, ModelDefinition{
			Name: "anthropic.claude-3-haiku-20240307-v1:0",
			Type: "chat",
		}, nil)
	})
}

func TestBedrockName(t *testing.T) {
	if got := (&Bedrock{}).Name(); got != provider {
		t.Fatalf("Name() = %q, want %q", got, provider)
	}
}

func TestDefineEmbedderRequiresInitializedPluginInstance(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	if got := b.DefineEmbedder(g, "amazon.titan-embed-text-v1"); got == nil {
		t.Fatal("DefineEmbedder returned nil for initialized plugin")
	}

	assertPanicsWith(t, "bedrock: Init not called", func() {
		(&Bedrock{}).DefineEmbedder(g, "amazon.titan-embed-text-v1")
	})
}

func TestModelLookupHelpers(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))
	modelName := "anthropic.claude-3-haiku-20240307-v1:0"

	if IsDefinedModel(g, modelName) {
		t.Fatal("IsDefinedModel returned true before model registration")
	}
	if got := Model(g, modelName); got != nil {
		t.Fatalf("Model returned %T before model registration, want nil", got)
	}

	defined := b.DefineModel(g, ModelDefinition{Name: modelName, Type: "chat"}, nil)
	if defined == nil {
		t.Fatal("DefineModel returned nil")
	}
	if !IsDefinedModel(g, modelName) {
		t.Fatal("IsDefinedModel returned false after model registration")
	}
	if got := Model(g, modelName); got == nil {
		t.Fatal("Model returned nil after model registration")
	}
}

func TestDefineCommonModelsRegistersExpectedAliases(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	models := DefineCommonModels(b, g)
	expectedAliases := []string{
		"claude-haiku",
		"claude-sonnet",
		"claude-opus-4",
		"claude-sonnet-4",
		"claude-3-7-sonnet",
		"nova-micro",
		"nova-lite",
		"nova-pro",
		"titan-text",
		"llama3-8b",
		"llama3-1-8b",
		"llama3-2-3b",
		"llama4-maverick",
		"llama4-scout",
		"deepseek-r1",
		"titan-image",
		"nova-canvas",
	}

	if len(models) != len(expectedAliases) {
		t.Fatalf("DefineCommonModels returned %d models, want %d", len(models), len(expectedAliases))
	}
	for _, alias := range expectedAliases {
		if models[alias] == nil {
			t.Fatalf("models[%q] is nil or missing", alias)
		}
	}
	if !IsDefinedModel(g, "anthropic.claude-3-haiku-20240307-v1:0") {
		t.Fatal("claude haiku model was not registered")
	}
	if !IsDefinedModel(g, "amazon.nova-canvas-v1:0") {
		t.Fatal("nova canvas image model was not registered")
	}
}

func TestDefineCommonEmbeddersRegistersExpectedAliases(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	embedders := DefineCommonEmbedders(b, g)
	expectedAliases := []string{
		"titan-embed",
		"titan-embed-v2",
		"titan-multimodal",
		"cohere-embed",
		"cohere-multilingual",
		"nova-embed",
	}

	if len(embedders) != len(expectedAliases) {
		t.Fatalf("DefineCommonEmbedders returned %d embedders, want %d", len(embedders), len(expectedAliases))
	}
	for _, alias := range expectedAliases {
		if embedders[alias] == nil {
			t.Fatalf("embedders[%q] is nil or missing", alias)
		}
	}
}

func TestInitUsesExplicitRegionAndDefaults(t *testing.T) {
	isolateAWSConfig(t)

	b := &Bedrock{Region: "us-west-2"}
	actions := b.Init(context.Background())

	if len(actions) != 0 {
		t.Fatalf("Init returned %d actions, want 0", len(actions))
	}
	if b.Region != "us-west-2" {
		t.Fatalf("Region = %q, want %q", b.Region, "us-west-2")
	}
	if b.MaxRetries != 3 {
		t.Fatalf("MaxRetries = %d, want 3", b.MaxRetries)
	}
	if b.RequestTimeout != 30*time.Second {
		t.Fatalf("RequestTimeout = %s, want 30s", b.RequestTimeout)
	}
	if !b.initted {
		t.Fatal("plugin was not marked initialized")
	}
	if b.client == nil {
		t.Fatal("client is nil")
	}
}

func TestInitUsesSDKRegionChain(t *testing.T) {
	isolateAWSConfig(t)
	t.Setenv("AWS_REGION", "eu-west-1")

	b := &Bedrock{}
	b.Init(context.Background())

	if b.Region != "" {
		t.Fatalf("Region = %q, want empty field when resolved by SDK chain", b.Region)
	}
	if !b.initted {
		t.Fatal("plugin was not marked initialized")
	}
	if b.client == nil {
		t.Fatal("client is nil")
	}
}

func TestInitPanicsWhenNoRegionResolved(t *testing.T) {
	isolateAWSConfig(t)

	assertPanicsContains(t, "no AWS region resolved", func() {
		(&Bedrock{}).Init(context.Background())
	})
}

func TestInitPanicsWhenAWSConfigHasNoRegion(t *testing.T) {
	b := &Bedrock{
		AWSConfig: &aws.Config{
			Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(
				"test-access-key",
				"test-secret-key",
				"",
			)),
		},
	}

	assertPanicsContains(t, "no AWS region resolved", func() {
		b.Init(context.Background())
	})
}

func TestDefineModelRegistersInferredCapabilityMetadata(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	m := b.DefineModel(g, ModelDefinition{
		Name: "us.unknown.model-v1:0",
		Type: "chat",
	}, nil)

	modelMeta := modelMetadata(t, m)
	if got := modelMeta["label"]; got != "bedrock-us.unknown.model-v1:0" {
		t.Fatalf("label = %v, want old default label", got)
	}
	if got := modelMeta["stage"]; got != ai.ModelStageUnstable {
		t.Fatalf("stage = %v, want %v", got, ai.ModelStageUnstable)
	}
	if modelMeta["customOptions"] == nil {
		t.Fatal("customOptions is nil, want generated Config schema")
	}

	supports, ok := modelMeta["supports"].(map[string]any)
	if !ok {
		t.Fatalf("supports = %T, want map[string]any", modelMeta["supports"])
	}
	for _, key := range []string{"tools", "toolChoice", "media", "multiturn", "systemRole"} {
		if got := supports[key]; got != true {
			t.Fatalf("supports[%q] = %v, want true", key, got)
		}
	}
	if got := supports["constrained"]; got != ai.ConstrainedSupportNone {
		t.Fatalf("supports[constrained] = %v, want %v", got, ai.ConstrainedSupportNone)
	}
}

func TestDefineModelRegistersProvidedMetadata(t *testing.T) {
	ctx := context.Background()
	b := testInitializedBedrock()
	g := genkit.Init(ctx, genkit.WithPlugins(b))

	customSchema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"custom": map[string]any{"type": "string"}},
	}
	m := b.DefineModel(g, ModelDefinition{
		Name: "anthropic.claude-3-haiku-20240307-v1:0",
		Type: "chat",
	}, &ai.ModelInfo{
		Label:        "Custom Bedrock Label",
		Stage:        ai.ModelStageDeprecated,
		ConfigSchema: customSchema,
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       false,
			Constrained: ai.ConstrainedSupportNone,
		},
		Versions: []string{"custom-version"},
	})

	modelMeta := modelMetadata(t, m)
	if got := modelMeta["label"]; got != "Custom Bedrock Label" {
		t.Fatalf("label = %v, want custom label", got)
	}
	if got := modelMeta["stage"]; got != ai.ModelStageDeprecated {
		t.Fatalf("stage = %v, want %v", got, ai.ModelStageDeprecated)
	}
	if got := modelMeta["customOptions"]; !reflect.DeepEqual(got, customSchema) {
		t.Fatalf("customOptions = %v, want provided schema", got)
	}

	versions, ok := modelMeta["versions"].([]string)
	if !ok {
		t.Fatalf("versions = %T, want []string", modelMeta["versions"])
	}
	if len(versions) != 1 || versions[0] != "custom-version" {
		t.Fatalf("versions = %v, want [custom-version]", versions)
	}
}

func testInitializedBedrock() *Bedrock {
	return &Bedrock{
		Region: "us-east-1",
		AWSConfig: &aws.Config{
			Region: "us-east-1",
			Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(
				"test-access-key",
				"test-secret-key",
				"",
			)),
		},
	}
}

func modelMetadata(t *testing.T, m ai.Model) map[string]any {
	t.Helper()
	action, ok := m.(api.Action)
	if !ok {
		t.Fatalf("model = %T, want api.Action", m)
	}
	modelMeta, ok := action.Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("metadata[model] = %T, want map[string]any", action.Desc().Metadata["model"])
	}
	return modelMeta
}

func assertPanicsWith(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		got := recover()
		if got == nil {
			t.Fatalf("expected panic %q, got none", want)
		}
		if got != want {
			t.Fatalf("panic = %v, want %q", got, want)
		}
	}()
	fn()
}

func assertPanicsContains(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		got := recover()
		if got == nil {
			t.Fatalf("expected panic containing %q, got none", want)
		}
		if !strings.Contains(fmt.Sprint(got), want) {
			t.Fatalf("panic = %v, want substring %q", got, want)
		}
	}()
	fn()
}

func isolateAWSConfig(t *testing.T) {
	t.Helper()
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config")
	credentialsPath := filepath.Join(tmpDir, "credentials")
	if err := os.WriteFile(configPath, []byte("[default]\n"), 0600); err != nil {
		t.Fatalf("failed to write test AWS config: %v", err)
	}
	if err := os.WriteFile(credentialsPath, []byte("[default]\naws_access_key_id = test-access-key\naws_secret_access_key = test-secret-key\n"), 0600); err != nil {
		t.Fatalf("failed to write test AWS credentials: %v", err)
	}

	t.Setenv("AWS_ACCESS_KEY_ID", "test-access-key")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
	t.Setenv("AWS_SESSION_TOKEN", "")
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_PROFILE", "default")
	t.Setenv("AWS_CONFIG_FILE", configPath)
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", credentialsPath)
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
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
			name:          "text/html produces DocumentBlock with html format",
			contentType:   "text/html",
			dataURL:       "data:text/html;base64," + txtB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatHtml,
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
			name:          "text/csv produces DocumentBlock with csv format",
			contentType:   "text/csv",
			dataURL:       "data:text/csv;base64," + txtB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatCsv,
		},
		{
			name:          "application/msword produces DocumentBlock with doc format",
			contentType:   "application/msword",
			dataURL:       "data:application/msword;base64," + pdfB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatDoc,
		},
		{
			name:          "DOCX produces DocumentBlock with docx format",
			contentType:   "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
			dataURL:       "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," + pdfB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatDocx,
		},
		{
			name:          "application/vnd.ms-excel produces DocumentBlock with xls format",
			contentType:   "application/vnd.ms-excel",
			dataURL:       "data:application/vnd.ms-excel;base64," + pdfB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatXls,
		},
		{
			name:          "XLSX produces DocumentBlock with xlsx format",
			contentType:   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
			dataURL:       "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + pdfB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatXlsx,
		},
		{
			name:          "MIME type with parameters strips params for matching",
			contentType:   "text/plain; charset=utf-8",
			dataURL:       "data:text/plain;charset=utf-8;base64," + txtB64,
			wantBlockType: "document",
			wantDocFormat: types.DocumentFormatTxt,
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
