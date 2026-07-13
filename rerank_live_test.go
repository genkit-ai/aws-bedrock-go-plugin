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
	"flag"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

var (
	testBedrockRerankModel = flag.String("test-bedrock-rerank-model", "", "Bedrock rerank model ID for live tests, for example cohere.rerank-v3-5:0")
)

func TestBedrockLive_CohereRerank(t *testing.T) {
	if *testRegion == "" {
		t.Skip("bedrock live tests skipped; pass -test-bedrock-region=<region>")
	}
	if *testBedrockRerankModel == "" {
		t.Skip("set -test-bedrock-rerank-model to run live Bedrock rerank test")
	}

	ctx := context.Background()
	bedrockPlugin := &Bedrock{
		Region: *testRegion,
	}
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	resp, err := Rerank(ctx, g, *testBedrockRerankModel, &ai.RerankerRequest{
		Query: ai.DocumentFromText("What is the capital of France?", nil),
		Documents: []*ai.Document{
			ai.DocumentFromText("The capital of France is Paris.", nil),
			ai.DocumentFromText("The tallest mountain is Everest.", nil),
			ai.DocumentFromText("Bananas are yellow.", nil),
		},
		Options: &RerankOptions{TopN: 2},
	})
	if err != nil {
		t.Fatalf("Rerank returned error: %v", err)
	}
	if len(resp.Documents) != 2 {
		t.Fatalf("Rerank returned %d documents, want 2", len(resp.Documents))
	}

	for i, doc := range resp.Documents {
		if len(doc.Content) == 0 || doc.Content[0].Text == "" {
			t.Fatalf("ranked document %d has no text content", i)
		}
		if doc.Metadata == nil {
			t.Fatalf("ranked document %d has nil metadata", i)
		}
	}
	if got := documentText(&ai.Document{Content: resp.Documents[0].Content}); !strings.Contains(got, "Paris") {
		t.Fatalf("top reranked document = %q, want Paris document", got)
	}
	if resp.Documents[0].Metadata.Score < resp.Documents[1].Metadata.Score {
		t.Fatalf("scores not descending: %f < %f", resp.Documents[0].Metadata.Score, resp.Documents[1].Metadata.Score)
	}
}
