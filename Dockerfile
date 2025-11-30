# Multi-stage build for claude-code-adapter
FROM golang:1.24-alpine AS builder

# Install git for fetching dependencies
RUN apk add --no-cache git

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o claude-code-adapter ./cmd/claude-code-adapter-cli

# Create a minimal runtime image
FROM alpine:3.21

# Install ca-certificates for HTTPS calls
RUN apk add --no-ca-certificates && update-ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S claude && \
    adduser -u 1001 -S claude -G claude

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/claude-code-adapter .

# Copy config template
COPY --from=builder /app/config.template.yaml ./config.template.yaml

# Create data directory for snapshots
RUN mkdir -p /app/data && chown -R claude:claude /app

# Switch to non-root user
USER claude

# Expose the default port (2194)
EXPOSE 2194

# Default environment variables
ENV ANTHROPIC_API_KEY=""
ENV OPENROUTER_API_KEY=""
ENV CONFIG_FILE="/app/config.yaml"

# Run the application with default config (will use env vars)
ENTRYPOINT ["./claude-code-adapter"]
CMD ["serve", "--host", "0.0.0.0"]
