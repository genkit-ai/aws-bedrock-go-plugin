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
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func TestRerankInvokesCohereRerankAndMapsScores(t *testing.T) {
	var gotBody cohereRerankRequest
	var gotPath string
	var gotContentType string
	var gotAccept string
	var handlerErr error

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.EscapedPath()
		gotContentType = r.Header.Get("Content-Type")
		gotAccept = r.Header.Get("Accept")

		body, err := io.ReadAll(r.Body)
		if err != nil {
			handlerErr = fmt.Errorf("failed to read request body: %w", err)
			http.Error(w, handlerErr.Error(), http.StatusInternalServerError)
			return
		}
		if err := json.Unmarshal(body, &gotBody); err != nil {
			handlerErr = fmt.Errorf("failed to unmarshal request body: %w\nbody: %s", err, string(body))
			http.Error(w, handlerErr.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprint(w, `{"results":[{"index":1,"relevance_score":0.94},{"index":0,"relevance_score":0.42}]}`); err != nil {
			t.Errorf("failed to write mock rerank response: %v", err)
		}
	}))
	defer server.Close()

	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:       "us-east-1",
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		HTTPClient:   server.Client(),
		BaseEndpoint: aws.String(server.URL),
	})

	doc1 := ai.DocumentFromText("first document", map[string]any{"id": "first"})
	doc2 := ai.DocumentFromText("second document", map[string]any{"id": "second"})
	resp, err := rerank(context.Background(), client, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
		Query:     ai.DocumentFromText("query text", nil),
		Documents: []*ai.Document{doc1, doc2},
		Options:   RerankOptions{TopN: 1},
	})
	if err != nil {
		t.Fatalf("rerank returned error: %v", err)
	}
	if handlerErr != nil {
		t.Fatal(handlerErr)
	}

	if !strings.Contains(gotPath, "/model/cohere.rerank-v3-5%3A0/invoke") {
		t.Fatalf("InvokeModel path = %q, want encoded model invoke path", gotPath)
	}
	if gotContentType != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", gotContentType)
	}
	if gotAccept != "application/json" {
		t.Fatalf("Accept = %q, want application/json", gotAccept)
	}
	if gotBody.Query != "query text" {
		t.Fatalf("query = %q, want query text", gotBody.Query)
	}
	if gotBody.APIVersion != cohereRerankAPIVersion {
		t.Fatalf("api_version = %d, want %d", gotBody.APIVersion, cohereRerankAPIVersion)
	}
	if gotBody.TopN != 1 {
		t.Fatalf("top_n = %d, want 1", gotBody.TopN)
	}
	if len(gotBody.Documents) != 2 || gotBody.Documents[0] != "first document" || gotBody.Documents[1] != "second document" {
		t.Fatalf("documents = %#v, want original document text order", gotBody.Documents)
	}

	if len(resp.Documents) != 2 {
		t.Fatalf("len(resp.Documents) = %d, want 2", len(resp.Documents))
	}
	if resp.Documents[0].Content[0].Text != "second document" {
		t.Fatalf("first ranked document text = %q, want second document", resp.Documents[0].Content[0].Text)
	}
	if resp.Documents[0].Metadata == nil || resp.Documents[0].Metadata.Score != 0.94 {
		t.Fatalf("first ranked score = %#v, want 0.94", resp.Documents[0].Metadata)
	}
	if resp.Documents[1].Content[0].Text != "first document" {
		t.Fatalf("second ranked document text = %q, want first document", resp.Documents[1].Content[0].Text)
	}
	if resp.Documents[1].Metadata == nil || resp.Documents[1].Metadata.Score != 0.42 {
		t.Fatalf("second ranked score = %#v, want 0.42", resp.Documents[1].Metadata)
	}
}

func TestRerankPublicWrapperValidation(t *testing.T) {
	ctx := context.Background()
	validReq := &ai.RerankerRequest{
		Query:     ai.DocumentFromText("query", nil),
		Documents: []*ai.Document{ai.DocumentFromText("document", nil)},
	}

	tests := []struct {
		name string
		g    *genkit.Genkit
		req  *ai.RerankerRequest
		want string
	}{
		{
			name: "nil genkit",
			g:    nil,
			req:  validReq,
			want: "Genkit instance required",
		},
		{
			name: "missing plugin",
			g:    genkit.Init(ctx),
			req:  validReq,
			want: "bedrock plugin not registered",
		},
		{
			name: "nil request",
			g:    genkit.Init(ctx, genkit.WithPlugins(testInitializedBedrock())),
			req:  nil,
			want: "request required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Rerank(ctx, tt.g, "cohere.rerank-v3-5:0", tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Rerank() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestRerankPublicWrapperValidationWithRegisteredPlugin(t *testing.T) {
	ctx := context.Background()
	validReq := &ai.RerankerRequest{
		Query:     ai.DocumentFromText("query", nil),
		Documents: []*ai.Document{ai.DocumentFromText("document", nil)},
	}

	t.Run("uninitialized plugin", func(t *testing.T) {
		b := testInitializedBedrock()
		g := genkit.Init(ctx, genkit.WithPlugins(b))
		b.mu.Lock()
		b.initted = false
		b.mu.Unlock()

		_, err := Rerank(ctx, g, "cohere.rerank-v3-5:0", validReq)
		if err == nil || !strings.Contains(err.Error(), "plugin not initialized") {
			t.Fatalf("Rerank() error = %v, want plugin not initialized", err)
		}
	})

	t.Run("empty model ID", func(t *testing.T) {
		g := genkit.Init(ctx, genkit.WithPlugins(testInitializedBedrock()))

		_, err := Rerank(ctx, g, "", validReq)
		if err == nil || !strings.Contains(err.Error(), "model ID required") {
			t.Fatalf("Rerank() error = %v, want model ID required", err)
		}
	})
}

func TestRerankPublicWrapperDelegatesToPluginClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var gotBody cohereRerankRequest
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("failed to unmarshal request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if gotBody.Query != "query text" {
			t.Errorf("query = %q, want query text", gotBody.Query)
		}
		if gotBody.TopN != 1 {
			t.Errorf("top_n = %d, want 1", gotBody.TopN)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"results":[{"index":0,"relevance_score":0.77}]}`)
	}))
	defer server.Close()

	b := &Bedrock{
		AWSConfig: &aws.Config{
			Region:       "us-east-1",
			Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
			HTTPClient:   server.Client(),
			BaseEndpoint: aws.String(server.URL),
		},
	}
	g := genkit.Init(context.Background(), genkit.WithPlugins(b))

	resp, err := Rerank(context.Background(), g, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
		Query:     ai.DocumentFromText("query text", nil),
		Documents: []*ai.Document{ai.DocumentFromText("document text", nil)},
		Options:   &RerankOptions{TopN: 1},
	})
	if err != nil {
		t.Fatalf("Rerank returned error: %v", err)
	}
	if len(resp.Documents) != 1 || resp.Documents[0].Metadata == nil || resp.Documents[0].Metadata.Score != 0.77 {
		t.Fatalf("response documents = %#v, want one scored document", resp.Documents)
	}
}

func TestRerankDefaultsTopNToDocumentCount(t *testing.T) {
	var gotBody cohereRerankRequest
	var handlerErr error

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			handlerErr = fmt.Errorf("failed to read request body: %w", err)
			http.Error(w, handlerErr.Error(), http.StatusInternalServerError)
			return
		}
		if err := json.Unmarshal(body, &gotBody); err != nil {
			handlerErr = fmt.Errorf("failed to unmarshal request body: %w", err)
			http.Error(w, handlerErr.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprint(w, `{"results":[]}`); err != nil {
			t.Errorf("failed to write mock rerank response: %v", err)
		}
	}))
	defer server.Close()

	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:       "us-east-1",
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		HTTPClient:   server.Client(),
		BaseEndpoint: aws.String(server.URL),
	})

	_, err := rerank(context.Background(), client, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
		Query: ai.DocumentFromText("query text", nil),
		Documents: []*ai.Document{
			ai.DocumentFromText("first document", nil),
			ai.DocumentFromText("second document", nil),
		},
		Options: map[string]any{"topN": float64(99)},
	})
	if err != nil {
		t.Fatalf("rerank returned error: %v", err)
	}
	if handlerErr != nil {
		t.Fatal(handlerErr)
	}

	if gotBody.TopN != 2 {
		t.Fatalf("top_n = %d, want document count 2", gotBody.TopN)
	}
}

func TestRerankEmptyDocumentsDoesNotInvokeModel(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:       "us-east-1",
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		HTTPClient:   server.Client(),
		BaseEndpoint: aws.String(server.URL),
	})

	resp, err := rerank(context.Background(), client, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
		Query: ai.DocumentFromText("query text", nil),
	})
	if err != nil {
		t.Fatalf("rerank returned error: %v", err)
	}
	if len(resp.Documents) != 0 {
		t.Fatalf("len(resp.Documents) = %d, want 0", len(resp.Documents))
	}
	if called {
		t.Fatal("InvokeModel should not be called for empty document list")
	}
}

func TestBuildRerankResponseRejectsOutOfRangeIndex(t *testing.T) {
	_, err := buildRerankResponse(cohereRerankResponse{
		Results: []cohereRerankResult{{Index: 1, RelevanceScore: 0.9}},
	}, []*ai.Document{ai.DocumentFromText("only document", nil)})
	if err == nil {
		t.Fatal("expected out-of-range index error")
	}
	if !strings.Contains(err.Error(), "result index 1 out of range") {
		t.Fatalf("error = %v, want out-of-range index", err)
	}
}

func TestBuildRerankResponseRejectsNilDocument(t *testing.T) {
	_, err := buildRerankResponse(cohereRerankResponse{
		Results: []cohereRerankResult{{Index: 0, RelevanceScore: 0.9}},
	}, []*ai.Document{nil})
	if err == nil {
		t.Fatal("expected nil document error")
	}
	if !strings.Contains(err.Error(), "references nil document") {
		t.Fatalf("error = %v, want nil document", err)
	}
}

func TestRerankValidationErrors(t *testing.T) {
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
	})

	tests := []struct {
		name string
		req  *ai.RerankerRequest
		want string
	}{
		{
			name: "nil request",
			req:  nil,
			want: "request required",
		},
		{
			name: "empty query",
			req: &ai.RerankerRequest{
				Query:     ai.DocumentFromText(" ", nil),
				Documents: []*ai.Document{ai.DocumentFromText("document", nil)},
			},
			want: "query has no text content",
		},
		{
			name: "empty document",
			req: &ai.RerankerRequest{
				Query:     ai.DocumentFromText("query", nil),
				Documents: []*ai.Document{ai.DocumentFromText("", nil)},
			},
			want: "document 0 has no text content",
		},
		{
			name: "unsupported options type",
			req: &ai.RerankerRequest{
				Query:     ai.DocumentFromText("query", nil),
				Documents: []*ai.Document{ai.DocumentFromText("document", nil)},
				Options:   "bad options",
			},
			want: "unsupported rerank options type string",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := rerank(context.Background(), client, "cohere.rerank-v3-5:0", tt.req)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestRerankOptions(t *testing.T) {
	tests := []struct {
		name    string
		input   any
		want    *RerankOptions
		wantErr string
	}{
		{
			name:  "nil options",
			input: nil,
			want:  nil,
		},
		{
			name:  "pointer options",
			input: &RerankOptions{TopN: 3},
			want:  &RerankOptions{TopN: 3},
		},
		{
			name:  "value options",
			input: RerankOptions{TopN: 5},
			want:  &RerankOptions{TopN: 5},
		},
		{
			name:  "map options",
			input: map[string]any{"topN": float64(7)},
			want:  &RerankOptions{TopN: 7},
		},
		{
			name:    "unsupported options",
			input:   "nonsense",
			wantErr: "unsupported rerank options type string",
		},
		{
			name: "invalid map options",
			input: map[string]any{
				"topN": make(chan int),
			},
			wantErr: "failed to marshal rerank options",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := rerankOptions(tt.input)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatal("expected error")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("rerankOptions returned error: %v", err)
			}
			if tt.want == nil {
				if got != nil {
					t.Fatalf("rerankOptions() = %+v, want nil", got)
				}
				return
			}
			if got == nil {
				t.Fatal("rerankOptions() = nil, want options")
			}
			if got.TopN != tt.want.TopN {
				t.Fatalf("TopN = %d, want %d", got.TopN, tt.want.TopN)
			}
		})
	}
}

func TestDocumentText(t *testing.T) {
	tests := []struct {
		name string
		doc  *ai.Document
		want string
	}{
		{
			name: "nil document",
			doc:  nil,
			want: "",
		},
		{
			name: "empty document",
			doc:  &ai.Document{},
			want: "",
		},
		{
			name: "ignores nil and non-text parts",
			doc: &ai.Document{
				Content: []*ai.Part{
					nil,
					ai.NewMediaPart("image/png", "data:image/png;base64,AAAA"),
					ai.NewDataPart("structured data"),
				},
			},
			want: "",
		},
		{
			name: "ignores whitespace-only text parts",
			doc: &ai.Document{
				Content: []*ai.Part{
					ai.NewTextPart(" "),
					ai.NewTextPart("\n\t"),
				},
			},
			want: "",
		},
		{
			name: "joins non-empty text parts",
			doc: &ai.Document{
				Content: []*ai.Part{
					ai.NewTextPart("first"),
					ai.NewMediaPart("image/png", "data:image/png;base64,AAAA"),
					ai.NewTextPart("second"),
				},
			},
			want: "first\nsecond",
		},
		{
			name: "trims final joined text",
			doc: &ai.Document{
				Content: []*ai.Part{
					ai.NewTextPart(" first "),
					ai.NewTextPart("second\n"),
				},
			},
			want: "first \nsecond",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := documentText(tt.doc)
			if got != tt.want {
				t.Fatalf("documentText() = %q, want %q", got, tt.want)
			}
		})
	}
}
