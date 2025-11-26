# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based Claude Code Adapter that acts as a proxy server, converting between Anthropic's Messages API format and OpenRouter's Chat Completions API format. It allows clients using the Anthropic API format to seamlessly work with OpenRouter providers or directly with Anthropic's API.

## Build and Development Commands

### Prerequisites
- Go 1.24 or later
- `defc` tool for code generation: `go install github.com/x5iu/defc@latest`

### Common Commands
- **Build**: `go build -o claude-code-adapter ./cmd/claude-code-adapter-cli`
- **Run Server**: `go run ./cmd/claude-code-adapter-cli serve`
- **Test All**: `go test ./...`
- **Format**: `go fmt ./...`
- **Vet**: `go vet ./...`
- **Generate Code**: `go generate ./...` (Run after modifying `pkg/provider/provider.go`)
- **Pre-commit Check**: `go fmt ./... && go vet ./... && go test ./...`

### Specific Test Commands
- **Verbose Tests**: `go test -v ./...`
- **Provider Tests**: `go test ./pkg/provider -v`
- **Adapter Tests**: `go test ./pkg/adapter -v`
- **Integration Tests** (Requires API keys):
  - OpenRouter: `OPENROUTER_API_KEY=your_key go test ./pkg/provider -v`
  - Anthropic: `ANTHROPIC_API_KEY=your_key go test ./pkg/provider -run TestGenerateAnthropicMessage -v`
- **Specific Scenarios**:
  - Reasoning Validation: `go test ./pkg/provider -run TestCreateOpenRouterChatCompletion_ReasoningValidation -v`
  - Stream Conversion: `go test ./pkg/adapter -run TestConvertOpenRouterStreamToAnthropicStream -v`

## Architecture

### Server Operation
The server (`cmd/claude-code-adapter-cli/serve.go`) operates as an HTTP proxy listening on port 2194 (default). It routes requests based on profile matching and handles format conversion.

### Core Components
- **Profile System** (`pkg/profile`): Model-to-configuration matching using prefix patterns (e.g., `claude-*`). First matching profile wins. See `config.template.yaml` for examples.
- **Provider Interface** (`pkg/provider`): Defines API interfaces using `defc` annotations. `provider_impl.go` is auto-generated.
- **Format Adapter** (`pkg/adapter/convert_request.go`): Converts Anthropic requests to OpenRouter format. Handles model mapping, tool calls, and thinking mode.
- **Stream Adapter** (`pkg/adapter/convert_stream.go`): Converts OpenRouter streaming responses to Anthropic streaming format. Handles event sequencing and content type transitions.
- **Data Types** (`pkg/datatypes`): Definitions for Anthropic (`pkg/datatypes/anthropic`) and OpenRouter (`pkg/datatypes/openrouter`) APIs.
- **Snapshot** (`pkg/snapshot`): Records request/response pairs to JSONL for debugging/auditing.

### Profile-Based Configuration
Profiles are defined as a map in `config.yaml`. Each profile specifies:
- `models`: List of patterns to match (supports `*` suffix for prefix matching)
- `provider`: Target provider (`openrouter` or `anthropic`)
- `options`, `anthropic`, `openrouter`: Provider-specific settings

Profile order in YAML determines matching priority. The `ProfileManager` iterates profiles in definition order and returns the first match.

### Code Generation
- Uses `github.com/x5iu/defc` to generate HTTP clients.
- **Workflow**: Modify `pkg/provider/provider.go` -> Run `go generate ./pkg/provider` -> Verify `provider_impl.go`.
- **Important**: Never manually edit `*_impl.go` files.

### Configuration
Configuration is managed via Viper with the following precedence:
1. Command-line flags (e.g., `--provider openrouter`)
2. Environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_BASE_URL`)
3. Config file (`config.yaml`)
4. Defaults

## Key Patterns & Notes
- **Provider Selection**: Automatically selects Anthropic provider if server tools (computer/bash) are present.
- **Reasoning Formats**: Supports `anthropic-claude-v1`, `openai-responses-v1`, and `google-gemini-v1`.
- **Error Handling**: Custom error types map provider errors to a unified format.
- **Snapshots**: Can record traffic via `--snapshot jsonl:path.jsonl`. **Warning**: May contain sensitive data.
