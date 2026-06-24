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

// Live embedding tests exercise real Bedrock embedding endpoints. They are
// skipped by default and only run when the required model flags are provided,
// so they never run in CI without explicit opt-in.
//
// Run individual families:
//
//	go test -run TestBedrockLive_TitanTextEmbedV1 \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-titan-text=amazon.titan-embed-text-v1
//
//	go test -run TestBedrockLive_TitanMultimodalEmbed \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-titan-multimodal=amazon.titan-embed-image-v1
//
//	go test -run TestBedrockLive_CohereTextEmbed \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-cohere-text=cohere.embed-english-v3
//
//	go test -run TestBedrockLive_CohereImageEmbed \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-cohere-text=cohere.embed-english-v3
//
//	go test -run TestBedrockLive_NovaTextEmbed \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-nova=amazon.nova-embed-text-v1:0
//
// Run the full embedding matrix at once:
//
//	go test -run 'TestBedrockLive_.*Embed' \
//	    -test-bedrock-region=us-east-1 \
//	    -test-bedrock-embed-titan-text=amazon.titan-embed-text-v1 \
//	    -test-bedrock-embed-titan-text-v2=amazon.titan-embed-text-v2:0 \
//	    -test-bedrock-embed-titan-multimodal=amazon.titan-embed-image-v1 \
//	    -test-bedrock-embed-cohere-text=cohere.embed-english-v3 \
//	    -test-bedrock-embed-cohere-multilingual=cohere.embed-multilingual-v3 \
//	    -test-bedrock-embed-nova=amazon.nova-embed-text-v1:0

import (
	"context"
	"flag"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// Embedding model flags — each defaults to "" (skip).
var (
	testEmbedTitanText       = flag.String("test-bedrock-embed-titan-text", "", "Titan text embedding model ID (e.g. amazon.titan-embed-text-v1)")
	testEmbedTitanTextV2     = flag.String("test-bedrock-embed-titan-text-v2", "", "Titan text embedding v2 model ID (e.g. amazon.titan-embed-text-v2:0)")
	testEmbedTitanMultimodal = flag.String("test-bedrock-embed-titan-multimodal", "", "Titan multimodal embedding model ID (e.g. amazon.titan-embed-image-v1)")
	testEmbedCohereText      = flag.String("test-bedrock-embed-cohere-text", "", "Cohere English embedding model ID (e.g. cohere.embed-english-v3)")
	testEmbedCohereMulti     = flag.String("test-bedrock-embed-cohere-multilingual", "", "Cohere multilingual embedding model ID (e.g. cohere.embed-multilingual-v3)")
	testEmbedNova            = flag.String("test-bedrock-embed-nova", "", "Nova text embedding model ID (e.g. amazon.nova-embed-text-v1:0)")
)

// minimal1x1PNG is a base64-encoded 1×1 white PNG image used in multimodal
// embedding live tests. It is the smallest valid PNG that Titan will accept.
const minimal1x1PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="

// requireLiveEmbed skips the test if the region or model flag is absent and
// returns an initialised Bedrock plugin with the named embedder registered.
func requireLiveEmbed(t *testing.T, modelID string) (context.Context, *genkit.Genkit, ai.Embedder) {
	t.Helper()
	if *testRegion == "" {
		t.Skip("embedding live tests skipped; pass -test-bedrock-region=<region>")
	}
	if modelID == "" {
		t.Skip("embedding live test skipped; pass the relevant -test-bedrock-embed-* flag")
	}
	ctx := context.Background()
	pb := &Bedrock{Region: *testRegion}
	g := genkit.Init(ctx, genkit.WithPlugins(pb))
	e := pb.DefineEmbedder(g, modelID)
	return ctx, g, e
}

// assertEmbeddings verifies that a response has the expected number of
// embeddings, each with a positive number of dimensions.
func assertEmbeddings(t *testing.T, resp *ai.EmbedResponse, wantCount int) {
	t.Helper()
	if resp == nil {
		t.Fatal("embed response is nil")
	}
	if len(resp.Embeddings) != wantCount {
		t.Fatalf("got %d embeddings, want %d", len(resp.Embeddings), wantCount)
	}
	for i, emb := range resp.Embeddings {
		if emb == nil {
			t.Fatalf("embeddings[%d] is nil", i)
		}
		if len(emb.Embedding) == 0 {
			t.Fatalf("embeddings[%d] has zero dimensions", i)
		}
	}
}

// ---- Titan text v1 ----------------------------------------------------------

func TestBedrockLive_TitanTextEmbedV1(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedTitanText)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs(
			"Machine learning is a subset of artificial intelligence.",
			"Natural language processing helps computers understand text.",
		),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 2)
	t.Logf("Titan text v1: %d dims per embedding", len(resp.Embeddings[0].Embedding))
}

// ---- Titan text v2 ----------------------------------------------------------

func TestBedrockLive_TitanTextEmbedV2(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedTitanTextV2)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs("Embeddings represent semantic meaning as vectors."),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 1)
	t.Logf("Titan text v2: %d dims", len(resp.Embeddings[0].Embedding))
}

// ---- Titan multimodal: text-only --------------------------------------------

func TestBedrockLive_TitanMultimodalEmbedText(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedTitanMultimodal)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs("A photograph of a mountain at sunset."),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 1)
	t.Logf("Titan multimodal (text): %d dims", len(resp.Embeddings[0].Embedding))
}

// ---- Titan multimodal: image ------------------------------------------------

func TestBedrockLive_TitanMultimodalEmbedImage(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedTitanMultimodal)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/png", "data:image/png;base64,"+minimal1x1PNG),
	}}
	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithDocs(doc),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 1)
	t.Logf("Titan multimodal (image): %d dims", len(resp.Embeddings[0].Embedding))
}

// ---- Titan multimodal: text + image -----------------------------------------

func TestBedrockLive_TitanMultimodalEmbedTextAndImage(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedTitanMultimodal)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewTextPart("a white pixel"),
		ai.NewMediaPart("image/png", "data:image/png;base64,"+minimal1x1PNG),
	}}
	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithDocs(doc),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 1)
	t.Logf("Titan multimodal (text+image): %d dims", len(resp.Embeddings[0].Embedding))
}

// ---- Cohere English text batch ----------------------------------------------

func TestBedrockLive_CohereTextEmbedEnglish(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedCohereText)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs(
			"The quick brown fox jumps over the lazy dog.",
			"Embedding models convert text to dense vectors.",
			"AWS Bedrock provides access to foundation models.",
		),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 3)
	t.Logf("Cohere English text: %d dims per embedding", len(resp.Embeddings[0].Embedding))

	// All three embeddings should come back ordered as inputs.
	for i := range resp.Embeddings {
		if len(resp.Embeddings[i].Embedding) != len(resp.Embeddings[0].Embedding) {
			t.Errorf("embeddings[%d] dims = %d, want %d (consistent dimensions)",
				i, len(resp.Embeddings[i].Embedding), len(resp.Embeddings[0].Embedding))
		}
	}
}

// ---- Cohere multilingual text -----------------------------------------------

func TestBedrockLive_CohereTextEmbedMultilingual(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedCohereMulti)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs(
			"Hello, how are you?",
			"Hola, ¿cómo estás?",
			"Bonjour, comment allez-vous?",
		),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 3)
	t.Logf("Cohere multilingual: %d dims per embedding", len(resp.Embeddings[0].Embedding))
}

// ---- Cohere image embedding -------------------------------------------------

func TestBedrockLive_CohereImageEmbed(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedCohereText)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/png", "data:image/png;base64,"+minimal1x1PNG),
	}}
	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithDocs(doc),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 1)
	t.Logf("Cohere image: %d dims", len(resp.Embeddings[0].Embedding))
}

// ---- Cohere: mixed text + image batch ---------------------------------------

func TestBedrockLive_CohereMixedBatch(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedCohereText)

	imgDoc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/png", "data:image/png;base64,"+minimal1x1PNG),
	}}
	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithDocs(
			ai.DocumentFromText("text document", nil),
			imgDoc,
			ai.DocumentFromText("another text", nil),
		),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 3)
	t.Logf("Cohere mixed batch: %d dims per embedding", len(resp.Embeddings[0].Embedding))
}

// ---- Nova text embedding ----------------------------------------------------

func TestBedrockLive_NovaTextEmbed(t *testing.T) {
	ctx, g, e := requireLiveEmbed(t, *testEmbedNova)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(e),
		ai.WithTextDocs(
			"Nova embedding model for semantic search.",
			"Dense vector representation of natural language.",
		),
	)
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	assertEmbeddings(t, resp, 2)
	t.Logf("Nova text: %d dims per embedding", len(resp.Embeddings[0].Embedding))
}
