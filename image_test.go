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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func TestGenerateImage_TitanUsesLatestUserPromptAndConfig(t *testing.T) {
	var gotBody map[string]any
	var gotPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"images":["titan-image-1","titan-image-2"]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("old prompt")}},
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("ignore model")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("new prompt"), ai.NewTextPart(" detail")}},
		},
		Config: map[string]any{
			"imageGenerationConfig": map[string]any{
				"height": 512,
				"seed":   42,
			},
		},
	}

	resp, err := b.generateImage(context.Background(), "amazon.titan-image-generator-v1", req, nil)
	if err != nil {
		t.Fatalf("generateImage error: %v", err)
	}
	if !strings.Contains(gotPath, "/model/amazon.titan-image-generator-v1/invoke") {
		t.Fatalf("InvokeModel path = %q, want Titan invoke path", gotPath)
	}
	params := gotBody["textToImageParams"].(map[string]any)
	if got := params["text"]; got != "new prompt detail" {
		t.Fatalf("prompt = %v, want latest user prompt", got)
	}
	cfg := gotBody["imageGenerationConfig"].(map[string]any)
	if got := cfg["height"]; got != float64(512) {
		t.Fatalf("height = %v, want 512", got)
	}
	if got := cfg["seed"]; got != float64(42) {
		t.Fatalf("seed = %v, want 42", got)
	}
	if got := cfg["width"]; got != float64(1024) {
		t.Fatalf("width = %v, want default 1024", got)
	}
	assertImageResponse(t, resp, req, "titan-image-1", "titan-image-2")
}

func TestGenerateImage_NovaCanvasKeepsNestedConfigOverride(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"images":["nova-image"]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	req := imagePromptRequest("paint a city")
	req.Config = map[string]any{
		"imageGenerationConfig": map[string]any{
			"quality": "premium",
			"width":   512,
		},
	}

	resp, err := b.generateImage(context.Background(), "amazon.nova-canvas-v1:0", req, nil)
	if err != nil {
		t.Fatalf("generateImage error: %v", err)
	}
	cfg := gotBody["imageGenerationConfig"].(map[string]any)
	if got := cfg["quality"]; got != "premium" {
		t.Fatalf("quality = %v, want premium", got)
	}
	if got := cfg["width"]; got != float64(512) {
		t.Fatalf("width = %v, want 512", got)
	}
	assertImageResponse(t, resp, req, "nova-image")
}

func TestGenerateImage_StableDiffusionXLUsesLegacyPayload(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"artifacts":[{"base64":"sdxl-image","finishReason":"SUCCESS"}]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	req := imagePromptRequest("legacy stability")

	resp, err := b.generateImage(context.Background(), "stability.stable-diffusion-xl-v1:0", req, nil)
	if err != nil {
		t.Fatalf("generateImage error: %v", err)
	}
	if got := gotBody["prompt"]; got != nil {
		t.Fatalf("prompt = %v, want absent for legacy Stable Diffusion", got)
	}
	prompts := gotBody["text_prompts"].([]any)
	first := prompts[0].(map[string]any)
	if got := first["text"]; got != "legacy stability" {
		t.Fatalf("text prompt = %v, want legacy stability", got)
	}
	assertImageResponse(t, resp, req, "sdxl-image")
}

func TestGenerateImage_ModernStabilityUsesPromptPayload(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"images":["modern-image"],"finish_reasons":["SUCCESS"]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	req := imagePromptRequest("modern stability")

	resp, err := b.generateImage(context.Background(), "stability.sd3-large-v1:0", req, nil)
	if err != nil {
		t.Fatalf("generateImage error: %v", err)
	}
	if got := gotBody["text_prompts"]; got != nil {
		t.Fatalf("text_prompts = %v, want absent for modern Stability", got)
	}
	if got := gotBody["prompt"]; got != "modern stability" {
		t.Fatalf("prompt = %v, want modern stability", got)
	}
	if got := gotBody["output_format"]; got != "png" {
		t.Fatalf("output_format = %v, want png", got)
	}
	assertImageResponse(t, resp, req, "modern-image")
}

func TestGenerateImage_ModernStabilityKeepsFixedDefaults(t *testing.T) {
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("unmarshal request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"images":["modern-image"],"finish_reasons":["SUCCESS"]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	req := imagePromptRequest("modern stability")
	req.Config = map[string]any{
		"aspect_ratio":  "16:9",
		"seed":          42,
		"output_format": "jpeg",
	}

	resp, err := b.generateImage(context.Background(), "stability.sd3-large-v1:0", req, nil)
	if err != nil {
		t.Fatalf("generateImage error: %v", err)
	}
	if got := gotBody["prompt"]; got != "modern stability" {
		t.Fatalf("prompt = %v, want modern stability", got)
	}
	if got := gotBody["aspect_ratio"]; got != nil {
		t.Fatalf("aspect_ratio = %v, want absent", got)
	}
	if got := gotBody["seed"]; got != nil {
		t.Fatalf("seed = %v, want absent", got)
	}
	if got := gotBody["output_format"]; got != "png" {
		t.Fatalf("output_format = %v, want fixed png", got)
	}
	assertImageResponse(t, resp, req, "modern-image")
}

func TestGenerateImage_ModernStabilityFinishReasonError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"images":["blocked-image"],"finish_reasons":["CONTENT_FILTERED"]}`)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	_, err := b.generateImage(context.Background(), "stability.stable-image-core-v1:0", imagePromptRequest("blocked"), nil)
	if err == nil || !strings.Contains(err.Error(), "CONTENT_FILTERED") {
		t.Fatalf("expected finish reason error, got %v", err)
	}
}

func TestGenerateImage_EmptyImageResponsesError(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		response string
	}{
		{
			name:     "titan",
			model:    "amazon.titan-image-generator-v1",
			response: `{"images":[]}`,
		},
		{
			name:     "legacy stable diffusion",
			model:    "stability.stable-diffusion-xl-v1:0",
			response: `{"artifacts":[]}`,
		},
		{
			name:     "modern stability",
			model:    "stability.stable-image-ultra-v1:0",
			response: `{"images":[]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, tt.response)
			}))
			defer server.Close()

			b := newTestBedrock(server)
			_, err := b.generateImage(context.Background(), tt.model, imagePromptRequest("prompt"), nil)
			if err == nil || !strings.Contains(err.Error(), "no images generated") {
				t.Fatalf("expected no images error, got %v", err)
			}
		})
	}
}

func TestGenerateImage_MissingPromptDoesNotInvokeModel(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()

	b := newTestBedrock(server)
	_, err := b.generateImage(context.Background(), "amazon.titan-image-generator-v1", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("system text")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewMediaPart("image/png", fakeImageDataURL)}},
		},
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "no text prompt") {
		t.Fatalf("expected no prompt error, got %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("InvokeModel calls = %d, want 0", calls.Load())
	}
}

// TestGenerateImage_ConfigOverridesSurviveGenkitSchemaValidation exercises the
// full genkit.Generate path (not the direct b.generateImage shortcut used by
// the other tests in this file). DefineModel attaches a ConfigSchema that
// genkit validates config against before our handler ever runs; the other
// tests can't catch a too-strict schema silently rejecting family-specific
// overrides like Stability's aspect_ratio or Titan's imageGenerationConfig.
func TestGenerateImage_ConfigOverridesSurviveGenkitSchemaValidation(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		config   map[string]any
		response string
	}{
		{
			name:     "titan nested imageGenerationConfig",
			model:    "amazon.titan-image-generator-v1",
			config:   map[string]any{"imageGenerationConfig": map[string]any{"seed": 7}},
			response: `{"images":["titan-image"]}`,
		},
		{
			name:     "nova canvas nested imageGenerationConfig",
			model:    "amazon.nova-canvas-v1:0",
			config:   map[string]any{"imageGenerationConfig": map[string]any{"quality": "premium"}},
			response: `{"images":["nova-image"]}`,
		},
		{
			name:     "legacy stable diffusion flat override",
			model:    "stability.stable-diffusion-xl-v1:0",
			config:   map[string]any{"cfg_scale": 10},
			response: `{"artifacts":[{"base64":"sdxl-image","finishReason":"SUCCESS"}]}`,
		},
		{
			name:     "modern stability fixed defaults with extra config",
			model:    "stability.sd3-large-v1:0",
			config:   map[string]any{"aspect_ratio": "16:9"},
			response: `{"images":["modern-image"],"finish_reasons":["SUCCESS"]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, tt.response)
			}))
			defer server.Close()

			ctx := context.Background()
			pb := &Bedrock{
				Region: "us-east-1",
				AWSConfig: &aws.Config{
					Region:       "us-east-1",
					Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
					HTTPClient:   server.Client(),
					BaseEndpoint: aws.String(server.URL),
				},
			}
			g := genkit.Init(ctx, genkit.WithPlugins(pb))
			m := pb.DefineModel(g, ModelDefinition{Name: tt.model, Type: "image"}, nil)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(m),
				ai.WithPrompt("a test prompt"),
				ai.WithConfig(tt.config),
			)
			if err != nil {
				t.Fatalf("Generate error: %v", err)
			}
			if resp.Message == nil || len(resp.Message.Content) == 0 {
				t.Fatal("expected at least one content part in response")
			}
		})
	}
}

func imagePromptRequest(prompt string) *ai.ModelRequest {
	return &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart(prompt)}},
		},
	}
}

func assertImageResponse(t *testing.T, resp *ai.ModelResponse, req *ai.ModelRequest, wantImages ...string) {
	t.Helper()
	if resp.Request != req {
		t.Fatalf("Request = %p, want %p", resp.Request, req)
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Fatalf("FinishReason = %v, want stop", resp.FinishReason)
	}
	if resp.Message == nil || resp.Message.Role != ai.RoleModel {
		t.Fatalf("Message = %#v, want model message", resp.Message)
	}
	if len(resp.Message.Content) != len(wantImages) {
		t.Fatalf("content length = %d, want %d", len(resp.Message.Content), len(wantImages))
	}
	for i, want := range wantImages {
		part := resp.Message.Content[i]
		if !part.IsMedia() {
			t.Fatalf("part %d is not media: %#v", i, part)
		}
		if part.ContentType != "image/png" {
			t.Fatalf("part %d content type = %q, want image/png", i, part.ContentType)
		}
		wantURL := "data:image/png;base64," + want
		if part.Text != wantURL {
			t.Fatalf("part %d data URL = %q, want %q", i, part.Text, wantURL)
		}
	}
}
