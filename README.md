# Claude Code Adapter

[![Go Version](https://img.shields.io/badge/go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/x5iu/claude-code-adapter)

A Go-based proxy server that seamlessly converts between Anthropic's Messages API format and OpenRouter's Chat Completions API format. This adapter allows clients using the Anthropic API format to work with both OpenRouter providers and Anthropic's API directly.

## Features

- **API Format Conversion**: Seamlessly converts between Anthropic Messages API and OpenRouter Chat Completions API
- **Multi-Provider Support**: Works with OpenRouter, Anthropic, and other providers
- **Profile-Based Configuration**: Define different configurations for different models using pattern matching
- **Streaming Support**: Full support for streaming responses from both APIs
- **Model Mapping**: Flexible model name mapping for OpenRouter compatibility
- **Pass-Through Mode**: Direct passthrough for Anthropic API when conversion isn't needed
- **Provider Preferences**: Configurable provider filtering for OpenRouter
- **Code Generation**: Automatic HTTP client generation using `defc`
- **Enhanced Logging**: Detailed request tracking with model and provider information
- **Request/Response Snapshots**: Record requests and responses to JSONL via `--snapshot`
- **Comprehensive Testing**: 87%+ test coverage with integration tests

## Quick Start

### Installation

#### Build from Source

```bash
# Clone the repository
git clone https://github.com/x5iu/claude-code-adapter.git
cd claude-code-adapter

# Build the CLI
go build -o claude-code-adapter ./cmd/claude-code-adapter-cli
```

#### Docker

```bash
# Build Docker image
docker build -t claude-code-adapter .

# Run with Docker (using environment variables)
docker run -d \
  -p 2194:2194 \
  -e ANTHROPIC_API_KEY="your_anthropic_key" \
  -e OPENROUTER_API_KEY="your_openrouter_key" \
  claude-code-adapter

# Run with custom config file
docker run -d \
  -p 2194:2194 \
  -v /path/to/config.yaml:/app/config.yaml \
  claude-code-adapter serve --config /app/config.yaml
```

##### Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  claude-code-adapter:
    build: .
    ports:
      - "2194:2194"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/app/data
    command: ["serve", "--host", "0.0.0.0", "--config", "/app/config.yaml"]
```

Create a `.env` file with your API keys:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Then run:

```bash
docker-compose up -d
```

### Basic Usage

```bash
# Start the proxy server (default port 2194)
./claude-code-adapter serve

# Run directly without building
go run ./cmd/claude-code-adapter-cli serve

# Start with custom port
./claude-code-adapter serve -p 8080

# Start with custom host
./claude-code-adapter serve --host 0.0.0.0

# Start with OpenRouter provider
./claude-code-adapter serve --provider openrouter

# Start with Anthropic provider
./claude-code-adapter serve --provider anthropic

# Enable debug logging
./claude-code-adapter serve --debug

# Enable pass-through mode for Anthropic (bypasses conversion)
./claude-code-adapter serve --enable-pass-through-mode

# Record request/response snapshots to JSONL
./claude-code-adapter serve --snapshot jsonl:./snapshots.jsonl

# Reasoning and behavior flags
./claude-code-adapter serve --strict
./claude-code-adapter serve --format anthropic-claude-v1
./claude-code-adapter serve --context-window-resize-factor 0.6
./claude-code-adapter serve --disable-interleaved-thinking
./claude-code-adapter serve --force-thinking

# Use custom config file
./claude-code-adapter serve -c ./config.yaml
# Config searched in: $HOME/.claude-code-adapter/config.yaml, ./config.yaml

# Show serve help
./claude-code-adapter serve --help
```

The server will listen on the configured address (default `127.0.0.1:2194`) and accept Anthropic Messages API requests at `/v1/messages`.

### Snapshots

Enable snapshot recording to a JSON Lines file using the `--snapshot` flag.

```bash
./claude-code-adapter serve --snapshot jsonl:./snapshots.jsonl
```

Notes:
- Format: JSON Lines; appends one record per request/response
- Default: disabled; enable only when needed
- WARNING: config.template.yaml enables snapshots for demonstration (snapshot: "jsonl:snapshot.jsonl"); set snapshot: "" or omit this key in your config.yaml to keep recording disabled
- Paths like jsonl:./snapshots.jsonl or jsonl:snapshots.jsonl are relative to the current working directory
- Security: snapshots may contain sensitive content; handle the file securely

## Configuration

The adapter can be configured through:
1. Command-line flags (highest priority)
2. Environment variables  
3. Configuration file (`config.yaml`)
4. Default values (lowest priority)

### Configuration File

Create a `config.yaml` file (see `config.template.yaml`):

```yaml
http:
  host: "127.0.0.1"     # Server host
  port: 2194            # Server port

# Profiles define configurations for different models
# Profile order matters - first matching profile wins
profiles:
  # Profile for Claude models using Anthropic provider
  anthropic-claude:
    models:
      - "claude-*"      # Matches all claude-* models
    provider: "anthropic"
    options:
      strict: false
      reasoning:
        format: "anthropic-claude-v1"
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      base_url: "https://api.anthropic.com"
    openrouter:
      api_key: "${OPENROUTER_API_KEY}"

  # Profile for OpenAI models via OpenRouter
  openrouter-openai:
    models:
      - "openai/*"
      - "gpt-*"
    provider: "openrouter"
    options:
      prevent_empty_text_tool_result: true
      reasoning:
        format: "openai-responses-v1"
        effort: "medium"
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
    openrouter:
      api_key: "${OPENROUTER_API_KEY}"
      base_url: "https://openrouter.ai/api"

  # Default catch-all profile
  default:
    models:
      - "*"             # Matches any model
    provider: "openrouter"
    options:
      context_window_resize_factor: 0.6
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
    openrouter:
      api_key: "${OPENROUTER_API_KEY}"
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

- **Profile System** (`pkg/profile/`): Model-to-configuration matching using prefix patterns (e.g., `claude-*`). First matching profile wins.
- **Provider Interface** (`pkg/provider/`): Main API interface with auto-generated HTTP client
- **Format Adapter** (`pkg/adapter/convert_request.go`): Converts Anthropic requests to OpenRouter format
- **Stream Adapter** (`pkg/adapter/convert_stream.go`): Converts OpenRouter streams to Anthropic format
- **Response Handler** (`pkg/provider/response_handler.go`): Handles streaming responses and error parsing
- **Data Types**: Complete type definitions for both Anthropic and OpenRouter APIs

### Server Operation

The adapter server operates as an HTTP proxy:

1. **Listens** on configured host/port (default `127.0.0.1:2194`) at endpoint `/v1/messages`
2. **Accepts** Anthropic Messages API requests from clients
3. **Matches** the request model against configured profiles to determine provider and settings
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