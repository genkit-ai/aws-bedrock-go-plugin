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
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
)

// ---- helpers ----------------------------------------------------------------

// newTestBedrock builds a Bedrock instance that routes InvokeModel through srv.
func newTestBedrock(srv *httptest.Server) *Bedrock {
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:       "us-east-1",
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		HTTPClient:   srv.Client(),
		BaseEndpoint: aws.String(srv.URL),
	})
	return &Bedrock{client: client, initted: true}
}

// titanTextResp returns a JSON response matching the Titan text embedding shape.
func titanTextResp(vec []float32) string {
	b, _ := json.Marshal(map[string]any{"embedding": vec})
	return string(b)
}

// cohereTypedResp returns a Cohere-typed embedding response (embedding_types).
func cohereTypedResp(vecs [][]float32) string {
	b, _ := json.Marshal(map[string]any{
		"id":            "test-id",
		"response_type": "embeddings_by_type",
		"embeddings":    map[string]any{"float": vecs},
	})
	return string(b)
}

// fakeImageBase64 is a trivially valid base64 string for test image data.
var fakeImageBase64 = base64.StdEncoding.EncodeToString([]byte("fake-png-bytes"))

// fakeImageDataURL wraps fakeImageBase64 in a data URL for use in ai.NewMediaPart.
var fakeImageDataURL = "data:image/png;base64," + fakeImageBase64

// ---- routing / validation ---------------------------------------------------

func TestEmbed_NilRequest(t *testing.T) {
	b := &Bedrock{initted: true}
	_, err := b.embed(context.Background(), "amazon.titan-embed-text-v1", nil)
	if err == nil || !strings.Contains(err.Error(), "request is nil") {
		t.Fatalf("expected nil-request error, got %v", err)
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	b := &Bedrock{initted: true}
	_, err := b.embed(context.Background(), "amazon.titan-embed-text-v1", &ai.EmbedRequest{})
	if err == nil || !strings.Contains(err.Error(), "no documents") {
		t.Fatalf("expected no-documents error, got %v", err)
	}
}

func TestEmbed_UnsupportedModel(t *testing.T) {
	b := &Bedrock{initted: true}
	_, err := b.embed(context.Background(), "unknown.model-v1:0", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("hello", nil)},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported embedding model") {
		t.Fatalf("expected unsupported-model error, got %v", err)
	}
}

func TestEmbed_EmptyDocumentErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("InvokeModel should not be called for empty documents")
	}))
	defer server.Close()
	b := newTestBedrock(server)

	tests := []struct {
		model string
		doc   *ai.Document
	}{
		{"amazon.titan-embed-text-v1", ai.DocumentFromText("", nil)},
		{"amazon.titan-embed-image-v1", ai.DocumentFromText("", nil)},
		{"cohere.embed-english-v3", ai.DocumentFromText("", nil)},
		{"amazon.nova-embed-text-v1:0", ai.DocumentFromText("", nil)},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			_, err := b.embed(context.Background(), tt.model, &ai.EmbedRequest{
				Input: []*ai.Document{tt.doc},
			})
			if err == nil {
				t.Fatalf("%s: expected error for empty document, got nil", tt.model)
			}
		})
	}
}

// ---- Titan text -------------------------------------------------------------

func TestEmbedTitanText_SingleDocument(t *testing.T) {
	want := []float32{0.1, 0.2, 0.3}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "amazon.titan-embed-text-v1", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("hello world", nil)},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	if len(resp.Embeddings[0].Embedding) != 3 {
		t.Fatalf("got %d dims, want 3", len(resp.Embeddings[0].Embedding))
	}
	if gotBody["inputText"] != "hello world" {
		t.Fatalf("inputText = %v, want hello world", gotBody["inputText"])
	}
}

func TestEmbedTitanText_MultipleDocumentsOrdered(t *testing.T) {
	var calls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp([]float32{0.1, 0.2}))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "amazon.titan-embed-text-v2:0", &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("first", nil),
			ai.DocumentFromText("second", nil),
			ai.DocumentFromText("third", nil),
		},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("got %d embeddings, want 3", len(resp.Embeddings))
	}
	for i, e := range resp.Embeddings {
		if e == nil {
			t.Fatalf("embeddings[%d] is nil", i)
		}
	}
	if calls.Load() != 3 {
		t.Fatalf("expected 3 InvokeModel calls, got %d", calls.Load())
	}
}

func TestEmbedTitanText_NilDocumentErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()
	b := newTestBedrock(server)

	_, err := b.embed(context.Background(), "amazon.titan-embed-text-v1", &ai.EmbedRequest{
		Input: []*ai.Document{nil},
	})
	if err == nil || !strings.Contains(err.Error(), "is nil") {
		t.Fatalf("expected nil-document error, got %v", err)
	}
}

// ---- Titan multimodal -------------------------------------------------------

func TestEmbedTitanMultimodal_TextOnly(t *testing.T) {
	want := []float32{0.9, 0.8}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "amazon.titan-embed-image-v1", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("text only", nil)},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0].Embedding) != 2 {
		t.Fatalf("unexpected embeddings: %v", resp.Embeddings)
	}
	if gotBody["inputText"] != "text only" {
		t.Fatalf("inputText = %v, want 'text only'", gotBody["inputText"])
	}
	if _, hasImg := gotBody["inputImage"]; hasImg {
		t.Fatal("inputImage should not be sent for text-only document")
	}
}

func TestEmbedTitanMultimodal_ImageOnly(t *testing.T) {
	want := []float32{0.7, 0.6}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/png", fakeImageDataURL),
	}}
	resp, err := b.embed(context.Background(), "amazon.titan-embed-image-v1", &ai.EmbedRequest{
		Input: []*ai.Document{doc},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	if gotBody["inputImage"] != fakeImageBase64 {
		t.Fatalf("inputImage = %v, want base64 data", gotBody["inputImage"])
	}
	if _, hasText := gotBody["inputText"]; hasText {
		t.Fatal("inputText should not be sent for image-only document")
	}
}

func TestEmbedTitanMultimodal_TextAndImage(t *testing.T) {
	want := []float32{0.5, 0.4}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewTextPart("describe this"),
		ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+fakeImageBase64),
	}}
	resp, err := b.embed(context.Background(), "amazon.titan-embed-image-v1", &ai.EmbedRequest{
		Input: []*ai.Document{doc},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	if gotBody["inputText"] != "describe this" {
		t.Fatalf("inputText = %v", gotBody["inputText"])
	}
	if gotBody["inputImage"] != fakeImageBase64 {
		t.Fatalf("inputImage mismatch")
	}
}

func TestEmbedTitanMultimodal_UnsupportedImageFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("InvokeModel should not be called for unsupported image format")
	}))
	defer server.Close()
	b := newTestBedrock(server)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/webp", "data:image/webp;base64,"+fakeImageBase64),
	}}
	_, err := b.embed(context.Background(), "amazon.titan-embed-image-v1", &ai.EmbedRequest{
		Input: []*ai.Document{doc},
	})
	if err == nil || !strings.Contains(err.Error(), "not supported by Titan") {
		t.Fatalf("expected unsupported-format error, got %v", err)
	}
}

func TestEmbedTitanMultimodal_NoContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("InvokeModel should not be called for empty document")
	}))
	defer server.Close()
	b := newTestBedrock(server)

	_, err := b.embed(context.Background(), "amazon.titan-embed-image-v1", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("", nil)},
	})
	if err == nil || !strings.Contains(err.Error(), "no text or image content") {
		t.Fatalf("expected no-content error, got %v", err)
	}
}

// ---- Cohere text batch ------------------------------------------------------

func TestEmbedCohere_TextBatch(t *testing.T) {
	want := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, cohereTypedResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "cohere.embed-english-v3", &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("first doc", nil),
			ai.DocumentFromText("second doc", nil),
		},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(resp.Embeddings))
	}
	// Both docs must be sent in ONE API call.
	texts, ok := gotBody["texts"].([]any)
	if !ok || len(texts) != 2 {
		t.Fatalf("expected 2 texts in batch, got body=%v", gotBody)
	}
	if gotBody["input_type"] != "search_document" {
		t.Fatalf("input_type = %v, want search_document", gotBody["input_type"])
	}
	// Ordering preserved.
	if resp.Embeddings[0].Embedding[0] != 0.1 {
		t.Fatalf("embeddings[0][0] = %v, want 0.1", resp.Embeddings[0].Embedding[0])
	}
	if resp.Embeddings[1].Embedding[0] != 0.3 {
		t.Fatalf("embeddings[1][0] = %v, want 0.3", resp.Embeddings[1].Embedding[0])
	}
}

func TestEmbedCohere_MultilingualTextBatch(t *testing.T) {
	want := [][]float32{{0.5, 0.6}}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, cohereTypedResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "cohere.embed-multilingual-v3", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("hola mundo", nil)},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
}

func TestEmbedCohere_ImageDocument(t *testing.T) {
	want := [][]float32{{0.7, 0.8}}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, cohereTypedResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	doc := &ai.Document{Content: []*ai.Part{
		ai.NewMediaPart("image/png", fakeImageDataURL),
	}}
	resp, err := b.embed(context.Background(), "cohere.embed-english-v3", &ai.EmbedRequest{
		Input: []*ai.Document{doc},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	// Must use image endpoint (input_type=image).
	if gotBody["input_type"] != "image" {
		t.Fatalf("input_type = %v, want image", gotBody["input_type"])
	}
	images, ok := gotBody["images"].([]any)
	if !ok || len(images) != 1 {
		t.Fatalf("expected 1 image, got body=%v", gotBody)
	}
	if images[0] != fakeImageBase64 {
		t.Fatalf("image base64 mismatch")
	}
}

func TestEmbedCohere_MixedTextAndImageDocumentsOrdered(t *testing.T) {
	// Documents: [image, text, image, text]
	// Expected result order preserved: [0]=img-emb, [1]=txt-emb, [2]=img-emb, [3]=txt-emb
	textEmbs := [][]float32{{0.1, 0.2}, {0.3, 0.4}}
	imgEmb := [][]float32{{0.9, 0.8}}

	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		if body["input_type"] == "search_document" {
			fmt.Fprint(w, cohereTypedResp(textEmbs))
		} else {
			fmt.Fprint(w, cohereTypedResp(imgEmb))
		}
		callCount++
	}))
	defer server.Close()
	b := newTestBedrock(server)

	imgDoc := &ai.Document{Content: []*ai.Part{ai.NewMediaPart("image/png", fakeImageDataURL)}}
	resp, err := b.embed(context.Background(), "cohere.embed-english-v3", &ai.EmbedRequest{
		Input: []*ai.Document{
			imgDoc,
			ai.DocumentFromText("text one", nil),
			imgDoc,
			ai.DocumentFromText("text two", nil),
		},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 4 {
		t.Fatalf("got %d embeddings, want 4", len(resp.Embeddings))
	}
	for i, e := range resp.Embeddings {
		if e == nil {
			t.Fatalf("embeddings[%d] is nil", i)
		}
	}
	// Image docs are at indices 0 and 2 → should have embedding 0.9.
	if resp.Embeddings[0].Embedding[0] != 0.9 {
		t.Fatalf("embeddings[0][0] = %v, want 0.9 (image)", resp.Embeddings[0].Embedding[0])
	}
	if resp.Embeddings[2].Embedding[0] != 0.9 {
		t.Fatalf("embeddings[2][0] = %v, want 0.9 (image)", resp.Embeddings[2].Embedding[0])
	}
	// Text docs are at indices 1 and 3 → from the text batch call.
	if resp.Embeddings[1].Embedding[0] != 0.1 {
		t.Fatalf("embeddings[1][0] = %v, want 0.1 (text[0])", resp.Embeddings[1].Embedding[0])
	}
	if resp.Embeddings[3].Embedding[0] != 0.3 {
		t.Fatalf("embeddings[3][0] = %v, want 0.3 (text[1])", resp.Embeddings[3].Embedding[0])
	}
}

// ---- Cohere response decoding -----------------------------------------------

func TestDecodeCohereEmbeddings_TypedFormat(t *testing.T) {
	vecs := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	body, _ := json.Marshal(map[string]any{
		"id":            "abc",
		"response_type": "embeddings_by_type",
		"embeddings":    map[string]any{"float": vecs},
	})
	got, err := decodeCohereEmbeddings(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(got))
	}
	if got[0][0] != 1.0 || got[1][0] != 3.0 {
		t.Fatalf("values mismatch: %v", got)
	}
}

func TestDecodeCohereEmbeddings_LegacyFormat(t *testing.T) {
	vecs := [][]float32{{5.0, 6.0}, {7.0, 8.0}}
	body, _ := json.Marshal(map[string]any{
		"id":            "abc",
		"response_type": "embeddings_floats",
		"embeddings":    vecs,
	})
	got, err := decodeCohereEmbeddings(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d embeddings, want 2", len(got))
	}
	if got[0][0] != 5.0 || got[1][0] != 7.0 {
		t.Fatalf("values mismatch: %v", got)
	}
}

func TestDecodeCohereEmbeddings_MalformedJSON(t *testing.T) {
	_, err := decodeCohereEmbeddings([]byte(`not json`))
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestDecodeCohereEmbeddings_MissingField(t *testing.T) {
	_, err := decodeCohereEmbeddings([]byte(`{"id":"abc"}`))
	if err == nil || !strings.Contains(err.Error(), "missing embeddings field") {
		t.Fatalf("expected missing-field error, got %v", err)
	}
}

// ---- Nova -------------------------------------------------------------------

func TestEmbedNova_SingleDocument(t *testing.T) {
	want := []float32{0.1, 0.2, 0.3}
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, titanTextResp(want))
	}))
	defer server.Close()
	b := newTestBedrock(server)

	resp, err := b.embed(context.Background(), "amazon.nova-embed-text-v1:0", &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("nova embed test", nil)},
	})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("got %d embeddings, want 1", len(resp.Embeddings))
	}
	if gotBody["inputText"] != "nova embed test" {
		t.Fatalf("inputText = %v, want 'nova embed test'", gotBody["inputText"])
	}
}

// ---- imageFromDocument ------------------------------------------------------

func TestImageFromDocument(t *testing.T) {
	tests := []struct {
		name        string
		doc         *ai.Document
		wantMIME    string
		wantBase64  string
	}{
		{
			name:       "nil document",
			doc:        nil,
			wantMIME:   "",
			wantBase64: "",
		},
		{
			name:       "text-only document",
			doc:        ai.DocumentFromText("hello", nil),
			wantMIME:   "",
			wantBase64: "",
		},
		{
			name: "PNG image",
			doc: &ai.Document{Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,"+fakeImageBase64),
			}},
			wantMIME:   "image/png",
			wantBase64: fakeImageBase64,
		},
		{
			name: "JPEG image",
			doc: &ai.Document{Content: []*ai.Part{
				ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+fakeImageBase64),
			}},
			wantMIME:   "image/jpeg",
			wantBase64: fakeImageBase64,
		},
		{
			name: "non-image media part is ignored",
			doc: &ai.Document{Content: []*ai.Part{
				ai.NewMediaPart("application/pdf", "data:application/pdf;base64,"+fakeImageBase64),
			}},
			wantMIME:   "",
			wantBase64: "",
		},
		{
			name: "first image wins",
			doc: &ai.Document{Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,"+fakeImageBase64),
				ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,aaaa"),
			}},
			wantMIME:   "image/png",
			wantBase64: fakeImageBase64,
		},
		{
			name: "text part before image",
			doc: &ai.Document{Content: []*ai.Part{
				ai.NewTextPart("some text"),
				ai.NewMediaPart("image/png", "data:image/png;base64,"+fakeImageBase64),
			}},
			wantMIME:   "image/png",
			wantBase64: fakeImageBase64,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMIME, gotBase64 := imageFromDocument(tt.doc)
			if gotMIME != tt.wantMIME {
				t.Errorf("mimeType = %q, want %q", gotMIME, tt.wantMIME)
			}
			if gotBase64 != tt.wantBase64 {
				t.Errorf("base64Data = %q, want %q", gotBase64, tt.wantBase64)
			}
		})
	}
}

// ---- isTitanSupportedImageMIME ----------------------------------------------

func TestIsTitanSupportedImageMIME(t *testing.T) {
	supported := []string{"image/jpeg", "image/jpg", "image/png"}
	unsupported := []string{"image/webp", "image/gif", "image/bmp", "image/tiff", "application/pdf", ""}

	for _, mime := range supported {
		if !isTitanSupportedImageMIME(mime) {
			t.Errorf("isTitanSupportedImageMIME(%q) = false, want true", mime)
		}
	}
	for _, mime := range unsupported {
		if isTitanSupportedImageMIME(mime) {
			t.Errorf("isTitanSupportedImageMIME(%q) = true, want false", mime)
		}
	}
}
