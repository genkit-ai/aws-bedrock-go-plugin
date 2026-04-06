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

// Package main demonstrates document (PDF) analysis with AWS Bedrock.
// The plugin routes document MIME types to Bedrock's DocumentBlock content
// type, which Claude models require for non-image files such as PDFs.
//
// Usage:
//
//	go run . document.pdf
package main

import (
	"context"
	"encoding/base64"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

func main() {
	ctx := context.Background()

	// Determine PDF path from args or use default.
	pdfPath := "document.pdf"
	if len(os.Args) > 1 {
		pdfPath = os.Args[1]
	}

	pdfBytes, err := os.ReadFile(pdfPath)
	if err != nil {
		log.Fatalf("Failed to read %q: %v\nPlease provide a PDF file path as an argument or place document.pdf in this directory.", pdfPath, err)
	}

	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1",
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	// Claude 3.7 Sonnet supports document inputs. For newer models (Haiku 4.5,
	// Sonnet 4, etc.) use their cross-region inference profile IDs, e.g.:
	//   "us.anthropic.claude-haiku-4-5-20251001-v1:0"
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Type: "chat",
	}, nil)

	// Pass the PDF as a data URI with MIME type application/pdf.
	// The plugin sends this as a DocumentBlock, which is required by Bedrock
	// for document types — sending it as an ImageBlock causes a 400 error.
	pdfDataURL := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfBytes)

	log.Printf("Sending %q to Bedrock for analysis...", pdfPath)

	response, err := genkit.Generate(ctx, g,
		ai.WithModel(claudeModel),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewMediaPart("application/pdf", pdfDataURL),
			ai.NewTextPart("Summarize this document in a few sentences."),
		)),
	)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	log.Printf("Response:\n%s", response.Text())
}
