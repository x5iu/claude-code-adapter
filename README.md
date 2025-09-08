# Claude Code Adapter

[![Go Version](https://img.shields.io/badge/go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/x5iu/claude-code-adapter)

A Go-based proxy server that seamlessly converts between Anthropic's Messages API format and OpenRouter's Chat Completions API format. This adapter allows clients using the Anthropic API format to work with both OpenRouter providers and Anthropic's API directly.

## Features

- **API Format Conversion**: Seamlessly converts between Anthropic Messages API and OpenRouter Chat Completions API
- **Multi-Provider Support**: Works with OpenRouter, Anthropic, and other providers
- **Streaming Support**: Full support for streaming responses from both APIs
- **Model Mapping**: Flexible model name mapping for OpenRouter compatibility
- **Pass-Through Mode**: Direct passthrough for Anthropic API when conversion isn't needed
- **Provider Preferences**: Configurable provider filtering for OpenRouter
- **Code Generation**: Automatic HTTP client generation using `defc`
- **Enhanced Logging**: Detailed request tracking with model and provider information
- **Comprehensive Testing**: 87%+ test coverage with integration tests

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/x5iu/claude-code-adapter.git
cd claude-code-adapter

# Build the CLI
go build -o claude-code-adapter ./cmd/claude-code-adapter-cli
```

### Basic Usage

```bash
# Start the proxy server (default port 2194)
./claude-code-adapter serve

# Start with custom port
./claude-code-adapter serve -p 8080

# Start with OpenRouter provider
./claude-code-adapter serve --provider openrouter

# Start with Anthropic provider
./claude-code-adapter serve --provider anthropic

# Enable debug logging
./claude-code-adapter serve --debug

# Enable pass-through mode for Anthropic (bypasses conversion)
./claude-code-adapter serve --enable-pass-through-mode

# Reasoning and behavior flags
./claude-code-adapter serve --strict
./claude-code-adapter serve --format anthropic-claude-v1
./claude-code-adapter serve --context-window-resize-factor 0.6
./claude-code-adapter serve --disable-interleaved-thinking
./claude-code-adapter serve --force-thinking

# Use custom config file
./claude-code-adapter serve -c ./config.yaml
```

The server will listen on `127.0.0.1:2194` and accept Anthropic Messages API requests at `/v1/messages`.

## Configuration

The adapter can be configured through:
1. Command-line flags (highest priority)
2. Environment variables  
3. Configuration file (`config.yaml`)
4. Default values (lowest priority)

### Configuration File

Create a `config.yaml` file (see `config.template.yaml`):

```yaml
provider: "openrouter"  # Default provider: openrouter or anthropic

http:
  port: 2194            # Server port

options:
  strict: false
  reasoning:
    format: "anthropic-claude-v1"   # or "openai-responses-v1"
    effort: "medium"                # minimal|low|medium|high
    delimiter: "/"                  # signature delimiter used in stream conversion
  models:                           # Model name mappings for OpenRouter
    claude-sonnet-4-20250514: "anthropic/claude-sonnet-4"
    claude-opus-4-1-20250805: "anthropic/claude-opus-4.1"
  context_window_resize_factor: 0.6
  disable_count_tokens_request: false

anthropic:
  enable_pass_through_mode: false
  disable_interleaved_thinking: false
  force_thinking: false
  base_url: "https://api.anthropic.com"
  version: "2023-06-01"

openrouter:
  base_url: "https://openrouter.ai/api"
  model_reasoning_format:
    anthropic/claude-sonnet-4: "anthropic-claude-v1"
    openai/gpt-5: "openai-responses-v1"
  allowed_providers:
    - "anthropic"
    - "google-vertex"
    - "amazon-bedrock"
```

### Environment Variables

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_VERSION="2023-06-01"
```

## Architecture

### Core Components

- **Provider Interface** (`pkg/provider/`): Main API interface with auto-generated HTTP client
- **Format Adapter** (`pkg/adapter/convert_request.go`): Converts Anthropic requests to OpenRouter format
- **Stream Adapter** (`pkg/adapter/convert_stream.go`): Converts OpenRouter streams to Anthropic format  
- **Response Handler** (`pkg/provider/response_handler.go`): Handles streaming responses and error parsing
- **Data Types**: Complete type definitions for both Anthropic and OpenRouter APIs

### Server Operation

The adapter server operates as an HTTP proxy:

1. **Listens** on `127.0.0.1:2194` at endpoint `/v1/messages`
2. **Accepts** Anthropic Messages API requests from clients  
3. **Routes** requests based on provider configuration and server tools
4. **Converts** between API formats when using OpenRouter
5. **Handles** both streaming and non-streaming responses
6. **Provides** detailed logging with request IDs and token usage tracking

## Development

### Prerequisites

- Go 1.24 or later
- `defc` tool for code generation:
  ```bash
  go install github.com/x5iu/defc@latest
  ```

### Building and Testing

```bash
# Build the project
go build ./...

# Run all tests
go test ./...

# Run tests with coverage
go test ./... -cover

# Run integration tests (requires API keys)
OPENROUTER_API_KEY=your_key go test ./pkg/provider -v
ANTHROPIC_API_KEY=your_key go test ./pkg/provider -run TestGenerateAnthropicMessage -v

# Generate code (after interface changes)
go generate ./...

# Format and vet code
go fmt ./... && go vet ./...

# Full pre-commit check
go fmt ./... && go vet ./... && go test ./...
```

### Code Generation

When modifying the provider interface in `pkg/provider/provider.go`:

1. Update the interface with proper `defc` annotations
2. Run `go generate ./pkg/provider` to regenerate client code
3. Test the changes with `go test ./pkg/provider -v`

**Important**: Never manually edit generated files (`*_impl.go`).

## API Examples

### Making a Request

Send a POST request to `http://127.0.0.1:2194/v1/messages` with Anthropic Messages API format:

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1000,
  "messages": [
    {
      "role": "user", 
      "content": "Hello, how are you?"
    }
  ]
}
```

The adapter will:
- Convert the request to OpenRouter format (if using OpenRouter provider)
- Map the model name according to configuration
- Handle the response conversion back to Anthropic format

### Streaming Requests

Add `"stream": true` to the request for streaming responses. The adapter maintains full compatibility with Anthropic's streaming event format.

## Testing

The project includes comprehensive tests:

- **Unit Tests**: Provider functionality and validation
- **Integration Tests**: Real API calls (requires API keys)
- **Adapter Tests**: Request/response conversion (87%+ coverage)
- **Stream Tests**: Streaming response conversion
- **Reasoning Tests**: Validation for thinking model outputs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`go test ./...`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [defc](https://github.com/x5iu/defc) for HTTP client code generation
- Uses [Cobra](https://github.com/spf13/cobra) for CLI functionality
- Configuration management via [Viper](https://github.com/spf13/viper)