# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Go-based proxy server converting between Anthropic's Messages API and OpenRouter's Chat Completions API. Supports both OpenRouter providers and direct Anthropic API access.

## Commands

### Prerequisites
- Go 1.24+
- `defc` for code generation: `go install github.com/x5iu/defc@latest`

### Build & Run
```bash
go build -o claude-code-adapter ./cmd/claude-code-adapter-cli
go run ./cmd/claude-code-adapter-cli serve          # Start server (port 2194)
go run ./cmd/claude-code-adapter-cli serve --debug  # With debug logging
```

### Test
```bash
go test ./...                                       # All tests
go test ./pkg/adapter -v                            # Adapter tests
go test ./pkg/provider -v                           # Provider tests
go test ./pkg/provider -run TestGenerateAnthropicMessage -v  # Single test
```

### Code Generation
```bash
go generate ./pkg/provider   # After modifying pkg/provider/provider.go
```
**Important**: Never manually edit `*_impl.go` files.

### Pre-commit
```bash
go fmt ./... && go vet ./... && go test ./...
```

## Architecture

### Request Flow
1. Server (`cmd/claude-code-adapter-cli/serve.go`) receives Anthropic Messages API request at `/v1/messages`
2. Profile System (`pkg/profile`) matches model to configuration (first match wins)
3. Format Adapter (`pkg/adapter/convert_request.go`) converts to OpenRouter format if needed
4. Provider (`pkg/provider`) sends request to upstream (OpenRouter or Anthropic)
5. Stream Adapter (`pkg/adapter/convert_stream.go`) converts response back to Anthropic format

### Core Components
| Package | Purpose |
|---------|---------|
| `pkg/profile` | Model-to-config matching via prefix patterns (`claude-*`) |
| `pkg/provider` | API interfaces with `defc` annotations; `provider_impl.go` is auto-generated |
| `pkg/adapter` | Bidirectional format conversion (request & stream) |
| `pkg/datatypes/anthropic` | Anthropic API types |
| `pkg/datatypes/openrouter` | OpenRouter API types |
| `pkg/snapshot` | Request/response recording to JSONL |

### Profile Configuration
Profiles in `config.yaml` specify: `models` (patterns), `provider` (openrouter/anthropic), and provider-specific settings. See `config.template.yaml` for full options.

### Configuration Precedence
1. CLI flags → 2. Environment variables → 3. `config.yaml` → 4. Defaults

## Key Behaviors
- **Auto Provider Selection**: Uses Anthropic provider when server tools (computer/bash) are present
- **Reasoning Formats**: `anthropic-claude-v1`, `openai-responses-v1`, `google-gemini-v1`
- **Snapshots**: `--snapshot jsonl:path.jsonl` records traffic (contains sensitive data)
