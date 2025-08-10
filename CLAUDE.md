# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based Claude Code Adapter that acts as a proxy server, converting between Anthropic's Messages API format and OpenRouter's Chat Completions API format. It allows clients using the Anthropic API format to seamlessly work with OpenRouter providers or directly with Anthropic's API. The project uses code generation via `github.com/x5iu/defc` to automatically generate HTTP client code from interface definitions.

## CLI Usage

### Building the CLI

```bash
# Build the CLI binary
go build -o claude-code-adapter ./cmd/claude-code-adapter-cli

# Build with version info
go build -ldflags "-X main.Version=v0.1.0" -o claude-code-adapter ./cmd/claude-code-adapter-cli
```

### Running the Server

```bash
# Start the proxy server with default settings (port 2194)
./claude-code-adapter serve

# Start with custom port
./claude-code-adapter serve -p 8080

# Enable debug logging
./claude-code-adapter serve --debug

# Use specific provider (openrouter or anthropic)
./claude-code-adapter serve --provider openrouter
./claude-code-adapter serve --provider anthropic

# Enable pass-through mode for Anthropic (bypasses conversion)
./claude-code-adapter serve --enable-pass-through-mode

# Use custom config file
./claude-code-adapter serve
# Config searched in: $HOME/.claude-code-adapter/config.yaml, ./config.yaml
```

### Configuration

The server can be configured via:
1. Command-line flags (highest priority)
2. Environment variables
3. Configuration file (config.yaml)
4. Default values (lowest priority)

Example config.yaml (see config.template.yaml):
```yaml
provider: "openrouter"  # Default provider: openrouter or anthropic
strict: false           # Strict validation mode
http:
  port: 2194           # Server port
mapping:
  models:              # Model name mappings for OpenRouter
    claude-sonnet-4-20250514: "anthropic/claude-sonnet-4"
    claude-opus-4-1-20250805: "anthropic/claude-opus-4.1"
anthropic:
  enable_pass_through_mode: false  # Pass requests directly without conversion
  base_url: "https://api.anthropic.com"
  version: "2023-06-01"
openrouter:
  base_url: "https://openrouter.ai/api"
  allowed_providers:   # Provider preference for OpenRouter
    - "anthropic"
    - "google-vertex"
    - "amazon-bedrock"
```

## Build and Development Commands

```bash
# Build the project
go build ./...

# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run specific test
go test ./pkg/provider -run TestNewProvider -v

# Run integration tests (requires OPENROUTER_API_KEY)
OPENROUTER_API_KEY=your_key go test ./pkg/provider -v

# Run reasoning validation test specifically
OPENROUTER_API_KEY=your_key go test ./pkg/provider -run TestCreateOpenRouterChatCompletion_ReasoningValidation -v

# Run provider preference test specifically  
OPENROUTER_API_KEY=your_key go test ./pkg/provider -run TestCreateOpenRouterChatCompletion_WithProviderPreference -v

# Run Anthropic API tests (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=your_key go test ./pkg/provider -run TestGenerateAnthropicMessage -v

# Run Anthropic thinking test specifically
ANTHROPIC_API_KEY=your_key go test ./pkg/provider -run TestGenerateAnthropicMessage_Thinking -v

# Run Anthropic token counting tests
ANTHROPIC_API_KEY=your_key go test ./pkg/provider -run TestCountAnthropicTokens -v

# Run adapter conversion tests
go test ./pkg/adapter -v

# Run specific adapter test
go test ./pkg/adapter -run TestConvertAnthropicRequestToOpenRouterRequest_BasicFields -v

# Run stream conversion tests
go test ./pkg/adapter -run TestConvertOpenRouterStreamToAnthropicStream -v

# Check test coverage
go test ./pkg/adapter -cover

# Check coverage across all packages
go test ./... -cover

# Generate code (when provider interface changes)
go generate ./...

# Format code
go fmt ./...

# Vet code for issues
go vet ./...

# Tidy dependencies
go mod tidy

# Run full pre-commit check (format, vet, test)
go fmt ./... && go vet ./... && go test ./...
```

## Architecture

### Server Operation

The adapter server (`cmd/claude-code-adapter-cli/serve.go`) operates as an HTTP proxy:
1. **Listens on** `127.0.0.1:2194` (configurable) at endpoint `/v1/messages`
2. **Accepts** Anthropic Messages API requests from clients
3. **Routes** requests based on:
   - Provider configuration (`openrouter` or `anthropic`)
   - Presence of server tools (forces Anthropic provider)
   - Pass-through mode setting
4. **Converts** between API formats when using OpenRouter:
   - Anthropic request → OpenRouter request (via adapter)
   - OpenRouter stream → Anthropic stream (via stream converter)
5. **Handles** both streaming and non-streaming responses
6. **Provides** detailed logging with request IDs and token usage tracking

Key features:
- Graceful shutdown with 15-second timeout
- Panic recovery with stack traces in debug mode
- Automatic retry logic for failed requests
- Token counting with 2-second timeout fallback
- Provider-specific headers (`X-Provider`, `X-Cc-Request-Id`)

### Core Components

- **Provider Interface** (`pkg/provider/provider.go`): Defines the main API interface using defc annotations for code generation
- **Generated Client** (`pkg/provider/provider.gen.go`): Auto-generated HTTP client implementation with retry logic and error handling
- **Response Handler** (`pkg/provider/response_handler.go`): Handles streaming responses and error parsing from both OpenRouter and Anthropic APIs
- **Format Adapter** (`pkg/adapter/convert.go`): Converts Anthropic API requests to OpenRouter format, handling message structure, tool calls, thinking mode, and content types
- **Stream Adapter** (`pkg/adapter/convert_stream.go`): Converts OpenRouter streaming responses to Anthropic streaming format, handling event sequence, content transitions, and usage tracking
- **OpenRouter Types** (`pkg/datatypes/openrouter/openrouter.go`): Complete type definitions for OpenRouter API including chat completions, streaming, and error types
- **Anthropic Types** (`pkg/datatypes/anthropic/anthropic.go`): Complete type definitions for Anthropic API including messages, events, thinking, and streaming

### Code Generation

The project uses `github.com/x5iu/defc` for HTTP client code generation. Key features:
- Interface methods with special comments generate HTTP client code
- Built-in retry logic (configured with `retry=2`)
- Automatic JSON encoding/decoding
- Streaming response support
- Configuration via Viper (`get_config` template function)

### Configuration

- Uses Viper for configuration management
- **OpenRouter**: API key expected at `openrouter.api_key`, base URL at `openrouter.base_url` (default: `https://openrouter.ai/api`)
- **Anthropic**: API key expected at `anthropic.api_key`, base URL at `anthropic.base_url`, version at `anthropic.version` (default: `2023-06-01`)
- Environment variables automatically bound via `option.go`:
  - `OPENROUTER_API_KEY` → `openrouter.api_key`
  - `OPENROUTER_BASE_URL` → `openrouter.base_url`
  - `ANTHROPIC_API_KEY` → `anthropic.api_key` 
  - `ANTHROPIC_BASE_URL` → `anthropic.base_url`
  - `ANTHROPIC_VERSION` → `anthropic.version`
- HTTP client automatically handles gzip compression
- Authorization: Bearer token for OpenRouter, X-API-Key for Anthropic

### Stream Processing

- **OpenRouter**: `ChatCompletionStream` implements Go 1.23 iterators (`iter.Seq2`) for processing streaming responses. Includes builders for accumulating streamed chunks into complete responses.
- **Anthropic**: `MessageStream` implements Go 1.23 iterators (`iter.Seq2`) for processing event-based streaming responses. Handles various event types including content deltas, thinking deltas, and message lifecycle events.

## Testing

### Test Structure

- **Unit Tests**: Basic provider functionality and validation
- **Adapter Tests**: Comprehensive tests for Anthropic-to-OpenRouter conversion and OpenRouter-to-Anthropic stream conversion (87%+ coverage)
- **OpenRouter Integration Tests**: End-to-end tests with real OpenRouter API calls  
- **Anthropic Integration Tests**: End-to-end tests with real Anthropic API calls
- **Reasoning Tests**: Specific validation for thinking model outputs (both OpenRouter and Anthropic)
- **Provider Preference Tests**: Validation of provider filtering using `WithProviderPreference`

### Test Requirements

- OpenRouter tests require `OPENROUTER_API_KEY` environment variable
- Anthropic tests require `ANTHROPIC_API_KEY` environment variable  
- Tests will skip automatically if required API keys are not provided
- OpenRouter reasoning tests check for `delta.Reasoning` content from thinking models
- Anthropic thinking tests check for `thinking_delta` events with thinking content

### Key Test Functions

**Basic Tests:**
- `TestNewProvider`: Basic provider instantiation

**OpenRouter Tests:**
- `TestCreateOpenRouterChatCompletion_ClaudeThinking`: End-to-end API call validation
- `TestCreateOpenRouterChatCompletion_ReasoningValidation`: Validates reasoning output from thinking models
- `TestCreateOpenRouterChatCompletion_WithProviderPreference`: Tests provider preference with "only anthropic" setting

**Anthropic Tests:**
- `TestGenerateAnthropicMessage_Basic`: Basic Anthropic API functionality
- `TestGenerateAnthropicMessage_Thinking`: Validates thinking output from Anthropic models
- `TestCountAnthropicTokens_Basic`: Basic token counting functionality
- `TestCountAnthropicTokens_WithSystem`: Token counting with system messages
- `TestCountAnthropicTokens_WithThinking`: Token counting with thinking mode enabled
- `TestCountAnthropicTokens_InvalidModel`: Error handling for invalid models
- `TestCountAnthropicTokens_EmptyMessages`: Error handling for empty message arrays

**Adapter Tests:**
- `TestConvertAnthropicRequestToOpenRouterRequest_BasicFields`: Tests basic field mapping
- `TestConvertAnthropicRequestToOpenRouterRequest_ToolChoice`: Tests all tool choice types
- `TestConvertAnthropicRequestToOpenRouterRequest_Messages`: Tests message content conversion
- `TestCanonicalOpenRouterMessages`: Tests message merging logic
- `TestConvertAnthropicToolResultMessageContentsToOpenRouterChatCompletionMessageContent`: Tests tool result conversion

**Stream Conversion Tests:**
- `TestConvertOpenRouterStreamToAnthropicStream_BasicConversion`: Tests basic stream conversion functionality
- `TestConvertOpenRouterStreamToAnthropicStream_ContentTypes`: Tests different content types (text, reasoning, tool calls)
- `TestConvertOpenRouterStreamToAnthropicStream_ContentTypeTransitions`: Tests transitions between content types
- `TestConvertOpenRouterStreamToAnthropicStream_FinishReasons`: Tests finish reason mapping
- `TestConvertOpenRouterStreamToAnthropicStream_Usage`: Tests usage tracking and token counting
- `TestConvertOpenRouterStreamToAnthropicStream_ErrorHandling`: Tests error propagation

## Code Generation Requirements

The project relies on `github.com/x5iu/defc` for code generation and requires **Go 1.24** or later.

### Prerequisites

1. **Install defc tool** (required for code generation):
   ```bash
   go install github.com/x5iu/defc@latest
   ```

2. **Verify Go version** (must be 1.24+):
   ```bash
   go version
   ```

### Code Generation Workflow

When modifying the provider interface:

1. **Modify interface** in `pkg/provider/provider.go` with proper defc annotations:
   - HTTP method and retry count: `// MethodName POST retry=2`
   - URL template: `{{ get_config "base_url" }}/endpoint`
   - Headers as comments: `// Header-Name: value`
   - Request body template: `// {{ json_encode .req }}`

2. **Regenerate code**:
   ```bash
   go generate ./pkg/provider
   # Or regenerate all:
   go generate ./...
   ```

3. **Verify generation**:
   - Check that `provider_impl.go` is updated
   - Run tests to ensure compatibility: `go test ./pkg/provider -v`
   - Never manually edit generated files (`*_impl.go`)

## Development Notes

- Always run `go generate` after modifying the provider interface
- The generated code (`provider_impl.go`) should not be manually edited
- Error handling follows OpenRouter API error format with custom error types
- All JSON marshaling/unmarshaling is handled automatically by the generated code
- The `NewChatCompletionBuilder()` function is exported for building complete responses from streams
- Use `lo.ToPtr()` for creating pointers to primitives instead of custom helper functions
- The adapter layer handles complex message merging via `canonicalOpenRouterMessages()` - be careful when modifying this function as it affects how multiple content blocks from the same Anthropic message are combined
- Stream conversion handles complex content type transitions (text → thinking → tool calls) and maintains proper event sequencing
- Both adapter and stream conversion tests cover edge cases including nil values, empty collections, and panic scenarios

## Key Architectural Patterns

### Provider Selection Logic
The server automatically selects the appropriate provider based on request characteristics:
- **Anthropic Provider**: Forced when server tools are present in the request (tool calls with type "computer" or "bash")
- **OpenRouter Provider**: Default for regular chat completions
- **Pass-Through Mode**: When enabled, Anthropic requests bypass conversion entirely

### Message Conversion Pipeline
1. **Request Conversion** (`pkg/adapter/convert_request.go`): Anthropic → OpenRouter format
   - Model name mapping via configuration
   - Message role and content transformation
   - Tool choice and tool definitions conversion
   - System message handling and merging
2. **Stream Conversion** (`pkg/adapter/convert_stream.go`): OpenRouter → Anthropic streaming format
   - Event-based streaming with proper sequencing
   - Content type transitions (text/reasoning/tool_use)
   - Usage tracking and token counting
   - Error propagation and finish reason mapping

### Error Handling Strategy
- **Custom Error Types**: Both OpenRouter and Anthropic error formats are parsed into structured error types
- **Retry Logic**: Built into the generated HTTP client via defc annotations
- **Panic Recovery**: Server-level panic recovery with stack traces in debug mode
- **Graceful Degradation**: Fallback behaviors for token counting and response processing

### Configuration Hierarchy
Configuration is resolved in order of precedence:
1. Command-line flags (highest)
2. Environment variables (auto-bound via `options.go`)
3. YAML configuration file (`config.yaml`)
4. Default values (lowest)

The `viper` configuration system automatically binds environment variables to nested config keys (e.g., `OPENROUTER_API_KEY` → `openrouter.api_key`).