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

// Live image-generation tests exercise real Bedrock image endpoints across
// model families. They are skipped by default and only run when the required
// model flags are provided, so they never run in CI without explicit opt-in.
//
// Run individual families:
//
//	go test -run TestBedrockLive_TitanImage \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-image-titan=amazon.titan-image-generator-v2:0
//
//	go test -run TestBedrockLive_NovaCanvasImage \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-image-nova-canvas=amazon.nova-canvas-v1:0
//
//	go test -run TestBedrockLive_StableDiffusionXLImage \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-image-sdxl=stability.stable-diffusion-xl-v1:0
//
//	go test -run TestBedrockLive_ModernStabilityImage \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-image-modern-stability=stability.sd3-large-v1:0
//
// Run the full image matrix at once:
//
//	go test -run 'TestBedrockLive_.*Image' \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-image-titan=amazon.titan-image-generator-v2:0 \
//	    -test-bedrock-image-nova-canvas=amazon.nova-canvas-v1:0 \
//	    -test-bedrock-image-sdxl=stability.stable-diffusion-xl-v1:0 \
//	    -test-bedrock-image-modern-stability=stability.sd3-large-v1:0

import (
	"context"
	"flag"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// Image model flags — each defaults to "" (skip).
var (
	testImageTitan           = flag.String("test-bedrock-image-titan", "", "Titan Image Generator model ID (e.g. amazon.titan-image-generator-v2:0)")
	testImageNovaCanvas      = flag.String("test-bedrock-image-nova-canvas", "", "Nova Canvas model ID (e.g. amazon.nova-canvas-v1:0)")
	testImageStableDiffusion = flag.String("test-bedrock-image-sdxl", "", "Legacy Stable Diffusion XL model ID (e.g. stability.stable-diffusion-xl-v1:0)")
	testImageModernStability = flag.String("test-bedrock-image-modern-stability", "", "Modern Stability model ID (e.g. stability.sd3-large-v1:0 or stability.stable-image-core-v1:0)")
)

// requireLiveImage skips the test if the region or model flag is absent and
// returns an initialised Bedrock plugin with the named image model registered.
func requireLiveImage(t *testing.T, modelID string) (context.Context, *genkit.Genkit, ai.Model) {
	t.Helper()
	if *testRegion == "" {
		t.Skip("image live tests skipped; pass -test-bedrock-region=<region>")
	}
	if modelID == "" {
		t.Skip("image live test skipped; pass the relevant -test-bedrock-image-* flag")
	}
	ctx := context.Background()
	pb := &Bedrock{Region: *testRegion}
	g := genkit.Init(ctx, genkit.WithPlugins(pb))
	m := pb.DefineModel(g, ModelDefinition{
		Name: modelID,
		Type: "image",
	}, nil)
	return ctx, g, m
}

// skipIfModelUnavailable reports whether err looks like Bedrock rejecting the
// model itself (invalid identifier, no account access, or a retired/legacy
// model your account/region no longer serves) rather than a real bug in the
// request. Live-test model availability varies per AWS account and region —
// treating this class of error as a skip (with the raw error logged) keeps
// the test informative without failing on infrastructure it doesn't control.
func skipIfModelUnavailable(t *testing.T, modelID string, err error) bool {
	t.Helper()
	msg := err.Error()
	unavailable := strings.Contains(msg, "ValidationException") && strings.Contains(msg, "model identifier is invalid")
	unavailable = unavailable || strings.Contains(msg, "AccessDeniedException")
	unavailable = unavailable || strings.Contains(msg, "ResourceNotFoundException")
	unavailable = unavailable || strings.Contains(msg, "marked by provider as Legacy")
	if !unavailable {
		return false
	}
	t.Logf("model %q unavailable in this account/region, skipping: %v", modelID, err)
	t.Skip("model unavailable; see log for the underlying Bedrock error")
	return true
}

// assertImageResponse verifies that a response has at least one image/png
// media part, each carrying non-empty base64 data.
func assertImageResponseLive(t *testing.T, resp *ai.ModelResponse) {
	t.Helper()
	if resp == nil || resp.Message == nil {
		t.Fatal("image response or message is nil")
	}
	var mediaParts int
	for i, part := range resp.Message.Content {
		if !part.IsMedia() {
			continue
		}
		mediaParts++
		if part.ContentType != "image/png" {
			t.Errorf("part %d content type = %q, want image/png", i, part.ContentType)
		}
		if !strings.HasPrefix(part.Text, "data:image/png;base64,") {
			t.Errorf("part %d data URL = %q, want data:image/png;base64,... prefix", i, part.Text)
		}
		if strings.TrimPrefix(part.Text, "data:image/png;base64,") == "" {
			t.Errorf("part %d has empty base64 payload", i)
		}
	}
	if mediaParts == 0 {
		t.Fatal("expected at least one image/png media part, got none")
	}
}

// ---- Titan Image Generator ---------------------------------------------------

func TestBedrockLive_TitanImage(t *testing.T) {
	ctx, g, m := requireLiveImage(t, *testImageTitan)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("A serene mountain landscape at sunset, painted in watercolor style"),
	)
	if err != nil {
		if skipIfModelUnavailable(t, *testImageTitan, err) {
			return
		}
		t.Fatalf("Generate error: %v", err)
	}
	assertImageResponseLive(t, resp)
}

// ---- Nova Canvas ---------------------------------------------------------

func TestBedrockLive_NovaCanvasImage(t *testing.T) {
	ctx, g, m := requireLiveImage(t, *testImageNovaCanvas)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("A futuristic city skyline at dawn, digital art"),
	)
	if err != nil {
		if skipIfModelUnavailable(t, *testImageNovaCanvas, err) {
			return
		}
		t.Fatalf("Generate error: %v", err)
	}
	assertImageResponseLive(t, resp)
}

// ---- Legacy Stable Diffusion XL --------------------------------------------

func TestBedrockLive_StableDiffusionXLImage(t *testing.T) {
	ctx, g, m := requireLiveImage(t, *testImageStableDiffusion)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("A cozy cabin in a snowy forest, oil painting style"),
	)
	if err != nil {
		if skipIfModelUnavailable(t, *testImageStableDiffusion, err) {
			return
		}
		t.Fatalf("Generate error: %v", err)
	}
	assertImageResponseLive(t, resp)
}

// ---- Modern Stability (sd3-*, stable-image-*) ------------------------------

func TestBedrockLive_ModernStabilityImage(t *testing.T) {
	ctx, g, m := requireLiveImage(t, *testImageModernStability)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("A vibrant coral reef teeming with fish, photorealistic"),
	)
	if err != nil {
		if skipIfModelUnavailable(t, *testImageModernStability, err) {
			return
		}
		t.Fatalf("Generate error: %v", err)
	}
	assertImageResponseLive(t, resp)
}

// TestBedrockLive_ModernStabilityImageWithConfig confirms per-request config
// overrides (e.g. output_format) reach modern Stability models.
func TestBedrockLive_ModernStabilityImageWithConfig(t *testing.T) {
	ctx, g, m := requireLiveImage(t, *testImageModernStability)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("A minimalist geometric pattern in blue and gold"),
		ai.WithConfig(map[string]any{
			"aspect_ratio": "16:9",
		}),
	)
	if err != nil {
		if skipIfModelUnavailable(t, *testImageModernStability, err) {
			return
		}
		t.Fatalf("Generate error: %v", err)
	}
	assertImageResponseLive(t, resp)
}
