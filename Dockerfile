# Multi-stage build for claude-code-adapter
# Build stage
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

# Build the application with optimizations
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o claude-code-adapter ./cmd/claude-code-adapter-cli

# Runtime stage
FROM alpine:3.21

# Install ca-certificates for HTTPS calls
RUN apk add --no-cache ca-certificates && update-ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S claude && \
    adduser -u 1001 -S claude -G claude

# Set working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/claude-code-adapter .

# Create data directory for snapshots
RUN mkdir -p /app/data && chown -R claude:claude /app

# Switch to non-root user
USER claude

# Expose the default port (2194)
EXPOSE 2194

# Default environment variables
ENV ANTHROPIC_API_KEY=""
ENV OPENROUTER_API_KEY=""

# Run the application
ENTRYPOINT ["./claude-code-adapter"]
CMD ["serve", "--host", "0.0.0.0"]
