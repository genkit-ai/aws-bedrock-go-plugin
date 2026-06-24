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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/firebase/genkit/go/ai"
)

// embed routes an embedding request to the appropriate model-family handler.
// Supported families:
//   - Amazon Titan Embed Image (titan-embed-image) — multimodal text + image
//   - Amazon Titan Embed Text (titan-embed-text) — text only
//   - Cohere Embed (cohere.embed-*) — batched text and per-document image
//   - Amazon Nova Embed (nova-embed) — text only
func (b *Bedrock) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("embed: request is nil")
	}
	if len(req.Input) == 0 {
		return nil, fmt.Errorf("embed: request contains no documents")
	}

	switch {
	case strings.Contains(modelName, "titan-embed-image"):
		return b.embedTitanMultimodal(ctx, modelName, req)
	case strings.Contains(modelName, "titan-embed"):
		return b.embedTitanText(ctx, modelName, req)
	case strings.Contains(modelName, "cohere"):
		return b.embedCohere(ctx, modelName, req)
	case strings.Contains(modelName, "nova-embed"):
		return b.embedNova(ctx, modelName, req)
	default:
		return nil, fmt.Errorf("embed: unsupported embedding model %q", modelName)
	}
}

// embedConcurrencyLimit caps the number of simultaneous InvokeModel calls to
// avoid AWS Bedrock ThrottlingException under large document batches.
const embedConcurrencyLimit = 10

// embedTitanText embeds documents using Amazon Titan text embedding models.
// Documents are processed concurrently; results are reassembled in original order.
func (b *Bedrock) embedTitanText(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	embeddings := make([]*ai.Embedding, len(req.Input))
	errs := make([]error, len(req.Input))
	var wg sync.WaitGroup
	sem := make(chan struct{}, embedConcurrencyLimit)

	for i, doc := range req.Input {
		wg.Add(1)
		go func(idx int, d *ai.Document) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			}
			if d == nil {
				errs[idx] = fmt.Errorf("embed: document %d is nil", idx)
				return
			}
			text := documentText(d)
			if text == "" {
				errs[idx] = fmt.Errorf("embed: document %d has no text content", idx)
				return
			}
			emb, err := b.getTitanTextEmbedding(ctx, modelName, text)
			if err != nil {
				errs[idx] = fmt.Errorf("embed: document %d: %w", idx, err)
				return
			}
			embeddings[idx] = &ai.Embedding{Embedding: emb}
		}(i, doc)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return &ai.EmbedResponse{Embeddings: embeddings}, nil
}

// embedTitanMultimodal embeds documents using the Amazon Titan multimodal
// embedding model (titan-embed-image-v1). Each document may contain text,
// image, or both; at least one must be present. Documents are processed
// concurrently; results are reassembled in original order.
//
// Titan multimodal only supports JPEG and PNG images.
func (b *Bedrock) embedTitanMultimodal(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	embeddings := make([]*ai.Embedding, len(req.Input))
	errs := make([]error, len(req.Input))
	var wg sync.WaitGroup
	sem := make(chan struct{}, embedConcurrencyLimit)

	for i, doc := range req.Input {
		wg.Add(1)
		go func(idx int, d *ai.Document) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			}
			if d == nil {
				errs[idx] = fmt.Errorf("embed: document %d is nil", idx)
				return
			}
			text := documentText(d)
			mime, imgBase64 := imageFromDocument(d)
			if text == "" && imgBase64 == "" {
				errs[idx] = fmt.Errorf("embed: document %d has no text or image content", idx)
				return
			}
			if imgBase64 != "" && !isTitanSupportedImageMIME(mime) {
				errs[idx] = fmt.Errorf("embed: document %d image format %q is not supported by Titan (use JPEG or PNG)", idx, mime)
				return
			}
			emb, err := b.getTitanMultimodalEmbedding(ctx, modelName, text, imgBase64)
			if err != nil {
				errs[idx] = fmt.Errorf("embed: document %d: %w", idx, err)
				return
			}
			embeddings[idx] = &ai.Embedding{Embedding: emb}
		}(i, doc)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return &ai.EmbedResponse{Embeddings: embeddings}, nil
}

// embedCohere embeds documents using a Cohere embedding model. Text documents
// are collected into a single batched request; image-only documents are
// processed concurrently one at a time. Results are reassembled in original
// order regardless of which batch each document belonged to.
func (b *Bedrock) embedCohere(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	type slot struct {
		idx     int
		content string // text or base64 image data
	}

	var textSlots []slot
	var imageSlots []slot

	for i, doc := range req.Input {
		if doc == nil {
			return nil, fmt.Errorf("embed: document %d is nil", i)
		}
		text := documentText(doc)
		_, imgBase64 := imageFromDocument(doc)
		switch {
		// Text takes priority: if a document has both text and an image, only
		// the text is sent. Cohere does not support mixed-modality per document.
		case text != "":
			textSlots = append(textSlots, slot{idx: i, content: text})
		case imgBase64 != "":
			imageSlots = append(imageSlots, slot{idx: i, content: imgBase64})
		default:
			return nil, fmt.Errorf("embed: document %d has no text or image content", i)
		}
	}

	embeddings := make([]*ai.Embedding, len(req.Input))

	// Batch text documents in API calls of up to 96 documents (Bedrock Cohere limit).
	const cohereTextBatchSize = 96
	for i := 0; i < len(textSlots); i += cohereTextBatchSize {
		end := min(i+cohereTextBatchSize, len(textSlots))
		chunk := textSlots[i:end]
		texts := make([]string, len(chunk))
		for j, s := range chunk {
			texts[j] = s.content
		}
		batch, err := b.getCohereTextEmbeddings(ctx, modelName, texts)
		if err != nil {
			return nil, fmt.Errorf("embed: Cohere text batch: %w", err)
		}
		if len(batch) != len(chunk) {
			return nil, fmt.Errorf("embed: Cohere returned %d text embeddings for %d inputs", len(batch), len(chunk))
		}
		for j, s := range chunk {
			embeddings[s.idx] = &ai.Embedding{Embedding: batch[j]}
		}
	}

	// Process image documents concurrently.
	if len(imageSlots) > 0 {
		imgEmbs := make([][]float32, len(imageSlots))
		imgErrs := make([]error, len(imageSlots))
		var wg sync.WaitGroup
		sem := make(chan struct{}, embedConcurrencyLimit)

		for i, s := range imageSlots {
			wg.Add(1)
			go func(batchIdx int, imgBase64 string) {
				defer wg.Done()
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-ctx.Done():
					imgErrs[batchIdx] = ctx.Err()
					return
				}
				batch, err := b.getCohereImageEmbeddings(ctx, modelName, []string{imgBase64})
				if err != nil {
					imgErrs[batchIdx] = err
					return
				}
				if len(batch) == 0 {
					imgErrs[batchIdx] = fmt.Errorf("cohere returned no embedding for image")
					return
				}
				imgEmbs[batchIdx] = batch[0]
			}(i, s.content)
		}
		wg.Wait()

		for i, err := range imgErrs {
			if err != nil {
				return nil, fmt.Errorf("embed: Cohere image document %d: %w", imageSlots[i].idx, err)
			}
		}
		for i, s := range imageSlots {
			embeddings[s.idx] = &ai.Embedding{Embedding: imgEmbs[i]}
		}
	}

	return &ai.EmbedResponse{Embeddings: embeddings}, nil
}

// embedNova embeds documents using an Amazon Nova text embedding model.
// Documents are processed concurrently; results are reassembled in original order.
func (b *Bedrock) embedNova(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	embeddings := make([]*ai.Embedding, len(req.Input))
	errs := make([]error, len(req.Input))
	var wg sync.WaitGroup
	sem := make(chan struct{}, embedConcurrencyLimit)

	for i, doc := range req.Input {
		wg.Add(1)
		go func(idx int, d *ai.Document) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			}
			if d == nil {
				errs[idx] = fmt.Errorf("embed: document %d is nil", idx)
				return
			}
			text := documentText(d)
			if text == "" {
				errs[idx] = fmt.Errorf("embed: document %d has no text content", idx)
				return
			}
			emb, err := b.getNovaEmbedding(ctx, modelName, text)
			if err != nil {
				errs[idx] = fmt.Errorf("embed: document %d: %w", idx, err)
				return
			}
			embeddings[idx] = &ai.Embedding{Embedding: emb}
		}(i, doc)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return &ai.EmbedResponse{Embeddings: embeddings}, nil
}

// getTitanTextEmbedding calls a Titan text embedding model for a single text.
func (b *Bedrock) getTitanTextEmbedding(ctx context.Context, modelName, text string) ([]float32, error) {
	body, err := json.Marshal(map[string]any{"inputText": text})
	if err != nil {
		return nil, err
	}
	callCtx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	resp, err := b.client.InvokeModel(callCtx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("InvokeModel: %w", err)
	}
	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("model returned an empty embedding vector")
	}
	return result.Embedding, nil
}

// getTitanMultimodalEmbedding calls the Titan multimodal embedding model.
// text and imgBase64 are both optional, but at least one must be non-empty.
// imgBase64 must be raw base64-encoded image bytes (JPEG or PNG only).
func (b *Bedrock) getTitanMultimodalEmbedding(ctx context.Context, modelName, text, imgBase64 string) ([]float32, error) {
	req := map[string]any{}
	if text != "" {
		req["inputText"] = text
	}
	if imgBase64 != "" {
		req["inputImage"] = imgBase64
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	callCtx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	resp, err := b.client.InvokeModel(callCtx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("InvokeModel: %w", err)
	}
	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("model returned an empty embedding vector")
	}
	return result.Embedding, nil
}

// getCohereTextEmbeddings sends a batched text embedding request to a Cohere
// embedding model and returns one embedding vector per input text, in order.
func (b *Bedrock) getCohereTextEmbeddings(ctx context.Context, modelName string, texts []string) ([][]float32, error) {
	body, err := json.Marshal(map[string]any{
		"texts":           texts,
		"input_type":      "search_document",
		"truncate":        "END",
		"embedding_types": []string{"float"},
	})
	if err != nil {
		return nil, err
	}
	callCtx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	resp, err := b.client.InvokeModel(callCtx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("InvokeModel: %w", err)
	}
	return decodeCohereEmbeddings(resp.Body)
}

// getCohereImageEmbeddings sends an image embedding request to a Cohere
// embedding model. images must be raw base64-encoded image bytes.
func (b *Bedrock) getCohereImageEmbeddings(ctx context.Context, modelName string, images []string) ([][]float32, error) {
	body, err := json.Marshal(map[string]any{
		"images":          images,
		"input_type":      "image",
		"embedding_types": []string{"float"},
	})
	if err != nil {
		return nil, err
	}
	callCtx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	resp, err := b.client.InvokeModel(callCtx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("InvokeModel: %w", err)
	}
	return decodeCohereEmbeddings(resp.Body)
}

// getNovaEmbedding calls an Amazon Nova text embedding model for a single text.
func (b *Bedrock) getNovaEmbedding(ctx context.Context, modelName, text string) ([]float32, error) {
	body, err := json.Marshal(map[string]any{"inputText": text})
	if err != nil {
		return nil, err
	}
	callCtx, cancel := b.withRequestTimeout(ctx)
	defer cancel()

	resp, err := b.client.InvokeModel(callCtx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelName),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("InvokeModel: %w", err)
	}
	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("model returned an empty embedding vector")
	}
	return result.Embedding, nil
}

// decodeCohereEmbeddings decodes a Cohere embedding response body into a slice
// of float32 vectors. It handles both the legacy flat format
// ({"embeddings": [[...], ...]}) and the newer typed format introduced by
// embedding_types ({"embeddings": {"float": [[...], ...]}}).
func decodeCohereEmbeddings(body []byte) ([][]float32, error) {
	var outer struct {
		Embeddings json.RawMessage `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &outer); err != nil {
		return nil, fmt.Errorf("parse Cohere response: %w", err)
	}
	if len(outer.Embeddings) == 0 {
		return nil, fmt.Errorf("cohere response missing embeddings field")
	}

	// Typed format: {"float": [[...], ...]}
	// Use TrimLeft to skip any leading whitespace before inspecting the first
	// meaningful byte — json.RawMessage preserves bytes verbatim from the
	// parent document, so leading whitespace is possible with pretty-printed
	// responses.
	trimmed := bytes.TrimLeft(outer.Embeddings, " \t\r\n")
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("cohere embeddings field is empty")
	}
	if trimmed[0] == '{' {
		var typed struct {
			Float [][]float32 `json:"float"`
		}
		if err := json.Unmarshal(outer.Embeddings, &typed); err != nil {
			return nil, fmt.Errorf("parse Cohere typed embeddings: %w", err)
		}
		if len(typed.Float) == 0 {
			return nil, fmt.Errorf("cohere typed response has no float embeddings")
		}
		return typed.Float, nil
	}

	// Legacy flat format: [[...], ...]
	var flat [][]float32
	if err := json.Unmarshal(outer.Embeddings, &flat); err != nil {
		return nil, fmt.Errorf("parse Cohere flat embeddings: %w", err)
	}
	if len(flat) == 0 {
		return nil, fmt.Errorf("cohere returned no embeddings")
	}
	return flat, nil
}

// imageFromDocument returns the MIME type and raw base64-encoded bytes of the
// first image media part found in doc. Returns empty strings if no image part
// is present.
func imageFromDocument(doc *ai.Document) (mimeType, base64Data string) {
	if doc == nil {
		return "", ""
	}
	for _, part := range doc.Content {
		if part == nil || !part.IsMedia() {
			continue
		}
		dataURL := part.Text
		mt := strings.ToLower(strings.TrimSpace(strings.SplitN(part.ContentType, ";", 2)[0]))
		// Fall back to the MIME type embedded in the data URL when ContentType is
		// absent — mirrors the same logic in generate.go's media-part handling.
		if mt == "" && strings.HasPrefix(dataURL, "data:") {
			if header, _, ok := strings.Cut(dataURL, ","); ok {
				mimeAndParams, _, _ := strings.Cut(header, ";")
				mt = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(mimeAndParams, "data:")))
			}
		}
		if !strings.HasPrefix(mt, "image/") {
			continue
		}
		// Strip the data URL header (data:image/png;base64,) to get raw base64.
		// If there is no comma the value is not a data URL; skip this part
		// rather than passing the whole string as base64 to the API.
		if _, after, ok := strings.Cut(dataURL, ","); ok {
			return mt, after
		}
	}
	return "", ""
}

// isTitanSupportedImageMIME reports whether a MIME type is accepted by the
// Amazon Titan multimodal embedding model (JPEG and PNG only).
func isTitanSupportedImageMIME(mimeType string) bool {
	switch mimeType {
	case "image/jpeg", "image/jpg", "image/png":
		return true
	default:
		return false
	}
}
