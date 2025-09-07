package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/samber/lo"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/x5iu/claude-code-adapter/pkg/adapter"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/provider"
	"github.com/x5iu/claude-code-adapter/pkg/utils"
)

const (
	ProviderAnthropic  = "anthropic"
	ProviderOpenRouter = "openrouter"
)

func newServeCommand() *cobra.Command {
	var configFile string
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start claude-code-adapter-cli http server",
		Args:  cobra.NoArgs,
		PreRun: func(*cobra.Command, []string) {
			if configFile != "" {
				viper.SetConfigFile(configFile)
			}
			if err := viper.ReadInConfig(); err != nil {
				if !errors.As(err, &viper.ConfigFileNotFoundError{}) {
					slog.Info(fmt.Sprintf("error reading config file: %s", err.Error()))
				}
				slog.Info("using default config")
			}
			viper.OnConfigChange(func(fsnotify.Event) {
				slog.Info("config file changed, reloading")
			})
			viper.WatchConfig()
			if viper.GetBool("debug") {
				slog.Info("using debug mode")
				slog.SetLogLoggerLevel(slog.LevelDebug)
				var debugBuf strings.Builder
				viper.DebugTo(&debugBuf)
				slog.Debug(">>>>>>>>>>>>>>>>> viper >>>>>>>>>>>>>>>>>" + "\n" + debugBuf.String())
				slog.Debug("<<<<<<<<<<<<<<<<< viper <<<<<<<<<<<<<<<<<")
			}
		},
		Run: serve,
	}
	flags := cmd.Flags()
	flags.StringVarP(&configFile, "config", "c", "", "config file (default is $HOME/.claude-code-adapter/config.yaml)")
	flags.Bool("debug", false, "enable debug logging")
	flags.Uint16P("port", "p", 2194, "port to serve on")
	flags.String("provider", "openrouter", "provider to use")
	flags.Bool("strict", false, "strict validation")
	flags.String("format", string(openrouter.ChatCompletionMessageReasoningDetailFormatAnthropicClaudeV1), "reasoning format")
	flags.Bool("enable-pass-through-mode", false, "enable pass through mode")
	flags.Float64("context-window-resize-factor", 1.0, "context window resize factor")
	flags.Bool("disable-interleaved-thinking", false, "disable interleaved thinking")
	cobra.CheckErr(viper.BindPFlag("debug", flags.Lookup("debug")))
	cobra.CheckErr(viper.BindPFlag("http.port", flags.Lookup("port")))
	cobra.CheckErr(viper.BindPFlag("provider", flags.Lookup("provider")))
	cobra.CheckErr(viper.BindPFlag("options.strict", flags.Lookup("strict")))
	cobra.CheckErr(viper.BindPFlag("options.reasoning.format", flags.Lookup("format")))
	cobra.CheckErr(viper.BindPFlag("options.context_window_resize_factor", flags.Lookup("context-window-resize-factor")))
	cobra.CheckErr(viper.BindPFlag("anthropic.enable_pass_through_mode", flags.Lookup("enable-pass-through-mode")))
	cobra.CheckErr(viper.BindPFlag("anthropic.disable_interleaved_thinking", flags.Lookup("disable-interleaved-thinking")))
	viper.SetOptions(viper.WithLogger(slog.Default()))
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("$HOME/.claude-code-adapter/")
	viper.AddConfigPath(".")
	return cmd
}

func serve(cmd *cobra.Command, _ []string) {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/messages", onMessages(cmd, provider.NewProvider()))
	server := &http.Server{
		Addr:     fmt.Sprintf("127.0.0.1:%d", viper.GetUint16("http.port")),
		Handler:  mux,
		ErrorLog: slog.NewLogLogger(slog.Default().Handler(), slog.LevelWarn),
	}
	slog.Info(fmt.Sprintf("starting http server, listening on %s", server.Addr))
	go server.ListenAndServe()
	<-ctx.Done()
	slog.Info("receive shutdown signal, shutting down http server")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error(fmt.Sprintf("error shutting down http server: %s", err.Error()))
		os.Exit(2)
	} else {
		slog.Info("http server is shutdown gracefully")
	}
}

func onMessages(cmd *cobra.Command, p provider.Provider) func(w http.ResponseWriter, r *http.Request) {
	var requestCounter atomic.Int64
	return func(w http.ResponseWriter, r *http.Request) {
		requestID := requestCounter.Add(1)
		r.Header.Del(anthropic.HeaderAPIKey)
		r.Header.Set("User-Agent", fmt.Sprintf("claude-code-adapter-cli/%s", cmd.Parent().Version[1:]))
		w.Header().Set("X-Cc-Request-Id", strconv.FormatInt(requestID, 10))
		defer func() {
			if err := recover(); err != nil {
				slog.Error(fmt.Sprintf("[%d] panic recovered: %v", requestID, err))
				slog.Debug(fmt.Sprintf(">>>>>>>>>>>>>>>>> [%d] stack >>>>>>>>>>>>>>>>>", requestID) + "\n" + string(utils.Stack()))
				slog.Debug(fmt.Sprintf("<<<<<<<<<<<<<<<<< [%d] stack <<<<<<<<<<<<<<<<<", requestID))
				respondError(w,
					http.StatusInternalServerError,
					fmt.Sprintf("An error occured while processing your request: %v", err),
				)
			}
		}()
		if !utils.IsContentType(r.Header, "application/json") {
			respondError(w,
				http.StatusBadRequest,
				fmt.Sprintf("Invalid Content-Type %q", r.Header.Get("Content-Type")),
			)
			return
		}
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			respondError(w,
				http.StatusInternalServerError,
				fmt.Sprintf("Failed to read request body: %s", err.Error()),
			)
			return
		}
		rawBody, _ = json.MarshalIndent(json.RawMessage(rawBody), "", "    ")
		slog.Debug(fmt.Sprintf(">>>>>>>>>>>>>>>>> [%d] anthropic request >>>>>>>>>>>>>>>>>", requestID) + "\n" + string(rawBody))
		slog.Debug(fmt.Sprintf("<<<<<<<<<<<<<<<<< [%d] anthropic request <<<<<<<<<<<<<<<<<", requestID))
		var req *anthropic.GenerateMessageRequest
		if err = json.Unmarshal(rawBody, &req); err != nil {
			respondError(w,
				http.StatusBadRequest,
				fmt.Sprintf("The request body is not valid JSON: %s", err.Error()),
			)
			return
		}
		r.Header.Del("Content-Type")
		r.Header.Del("Content-Length")
		r.Header.Del("Transfer-Encoding")
		r.Header.Del("Accept-Encoding")
		slog.Info(fmt.Sprintf("[%d] request model: %s", requestID, req.Model))
		var (
			inputTokens  int64
			outputTokens int64
			stopReason   = anthropic.StopReason("unknown")
		)
		countTokensCtx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		// When tool.type is null, the /v1/messages/count_tokens endpoint will return the error
		// tools.0.get_defaulted_tool_discriminator(): Field required, so we need to preprocess tools to avoid this error.
		for _, tool := range req.Tools {
			if tool.Type == nil {
				tool.Type = lo.ToPtr(anthropic.ToolTypeCustom)
			}
		}
		if !viper.GetBool("options.disable_count_tokens_request") {
			usage, err := p.CountAnthropicTokens(countTokensCtx, &anthropic.CountTokensRequest{
				System:     req.System,
				Model:      req.Model,
				Messages:   req.Messages,
				Thinking:   req.Thinking,
				ToolChoice: req.ToolChoice,
				Tools:      req.Tools,
			})
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					slog.Warn(fmt.Sprintf("[%d] token calculation timed out", requestID))
				} else {
					slog.Error(fmt.Sprintf("[%d] error making CountAnthropicTokens request: %s", requestID, err.Error()))
				}
			} else {
				inputTokens = usage.InputTokens
				slog.Info(fmt.Sprintf("[%d] request input tokens (estimated): %d", requestID, inputTokens))
			}
		}
		hasServerTools := func() bool {
			return lo.ContainsBy(req.Tools, func(tool *anthropic.Tool) bool {
				isServerTool := tool.Type != nil && *tool.Type != anthropic.ToolTypeCustom
				if isServerTool {
					slog.Info(fmt.Sprintf("[%d] request contains %s tool: %s", requestID, *tool.Type, tool.Name))
				}
				return isServerTool
			})
		}
		useInterleavedThinking := func() bool {
			budgetTokensOverflow := req.Thinking != nil && req.Thinking.Type == anthropic.ThinkingTypeEnabled && req.Thinking.BudgetTokens >= req.MaxTokens
			if budgetTokensOverflow {
				var hasInterleavedFeatures bool
				for _, features := range r.Header.Values(anthropic.HeaderBeta) {
					if features != "" {
						for _, feature := range strings.Split(features, ",") {
							if strings.EqualFold(strings.TrimSpace(feature), anthropic.BetaFeatureInterleavedThinking20250514) {
								hasInterleavedFeatures = true
								break
							}
						}
					}
				}
				if !hasInterleavedFeatures {
					r.Header.Add(anthropic.HeaderBeta, anthropic.BetaFeatureInterleavedThinking20250514)
				}
				slog.Info(fmt.Sprintf("[%d] request's thinking tokens (%d) is greater than max tokens (%d)", requestID, req.Thinking.BudgetTokens, req.MaxTokens))
			}
			return budgetTokensOverflow && !viper.GetBool("anthropic.disable_interleaved_thinking")
		}
		useAnthropicProvider := func(provider string) bool {
			return hasServerTools() || useInterleavedThinking() || provider == ProviderAnthropic
		}
		var (
			stream     anthropic.MessageStream
			ccProvider = viper.GetString("provider")
			orProvider = "<unknown>"
		)
		defer func() {
			if stream != nil {
				for range stream {
				}
			}
		}()
		if useAnthropicProvider(ccProvider) {
			slog.Info(fmt.Sprintf("[%d] using provider %q", requestID, ProviderAnthropic))
			w.Header().Set("X-Provider", ProviderAnthropic)
			var (
				header                http.Header
				reader                io.ReadCloser
				enablePassThroughMode = viper.GetBool("anthropic.enable_pass_through_mode")
			)
			if enablePassThroughMode {
				reader, header, err = p.MakeAnthropicMessagesRequest(r.Context(),
					utils.NewResettableReader(rawBody),
					provider.WithQuery("beta", "true"),
					provider.WithHeaders(r.Header),
				)
			} else {
				stream, header, err = p.GenerateAnthropicMessage(r.Context(), req,
					provider.WithQuery("beta", "true"),
					provider.WithHeaders(r.Header),
					provider.ReplaceBody(rawBody),
				)
			}
			for k, vv := range header {
				for _, v := range vv {
					w.Header().Add(k, v)
				}
			}
			if err != nil {
				slog.Error(fmt.Sprintf("[%d] error making anthropic /v1/messages request: %s", requestID, err.Error()))
				if providerError, isProviderError := provider.ParseError(err); isProviderError {
					respondError(w, providerError.StatusCode(), providerError.Message())
				} else {
					respondError(w, http.StatusInternalServerError, err.Error())
				}
				return
			}
			if enablePassThroughMode {
				defer reader.Close()
				recvBytes, err := io.ReadAll(reader)
				if err != nil {
					slog.Error(fmt.Sprintf("[%d] error reading anthropic /v1/messages response: %s", requestID, err))
					respondError(w, http.StatusInternalServerError, err.Error())
				} else {
					slog.Debug(fmt.Sprintf(">>>>>>>>>>>>>>>>> [%d] anthropic response >>>>>>>>>>>>>>>>>", requestID) + "\n" + string(recvBytes))
					slog.Debug(fmt.Sprintf("<<<<<<<<<<<<<<<<< [%d] anthropic response <<<<<<<<<<<<<<<<<", requestID))
					if _, err = w.Write(recvBytes); err != nil {
						slog.Warn(fmt.Sprintf("[%d] error sending Anthropic response: %s", requestID, err))
					}
				}
				return
			}
		} else {
			switch ccProvider {
			case ProviderOpenRouter:
				fallthrough
			default:
				ccProvider = ProviderOpenRouter
				slog.Info(fmt.Sprintf("[%d] using provider %q", requestID, ProviderOpenRouter))
				w.Header().Set("X-Provider", ProviderOpenRouter)
				allowedProviders := viper.GetStringSlice("openrouter.allowed_providers")
				orStream, _, err := p.CreateOpenRouterChatCompletion(
					r.Context(),
					adapter.ConvertAnthropicRequestToOpenRouterRequest(req),
					openrouter.WithIdentity("https://github.com/x5iu/claude-code-adapter", "claude-code-adapter"),
					openrouter.WithProviderPreference(&openrouter.ProviderPreference{
						Order:             allowedProviders,
						AllowFallbacks:    lo.ToPtr(true),
						RequireParameters: lo.ToPtr(false), // OpenRouter does not support all Anthropic parameters.
						Only:              allowedProviders,
						Sort:              lo.ToPtr(openrouter.ProviderSortMethodThroughput),
					}),
				)
				if err != nil {
					slog.Error(fmt.Sprintf("[%d] error making OpenRouter ChatCompletions request: %s", requestID, err.Error()))
					if providerError, isProviderError := provider.ParseError(err); isProviderError {
						respondError(w, providerError.StatusCode(), providerError.Message())
					} else {
						respondError(w, http.StatusInternalServerError, err.Error())
					}
					return
				}
				stream = adapter.ConvertOpenRouterStreamToAnthropicStream(
					orStream,
					adapter.WithInputTokens(inputTokens),
					adapter.ExtractOpenRouterProvider(&orProvider),
				)
			}
		}
		dstMessageBuilder := anthropic.NewMessageBuilder()
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(http.StatusOK)
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		for event, err := range stream {
			if err != nil {
				if req.Stream {
					slog.Error(fmt.Sprintf("[%d] error transfering response stream: %s", requestID, err.Error()))
					fmt.Fprintf(w, "event: %s\n", anthropic.EventTypeError)
					fmt.Fprintf(w, "data: %s\n\n", utils.JSONEncodeString(&anthropic.StreamError{
						ErrType:    anthropic.ErrorContentType,
						ErrMessage: err.Error(),
					}))
				} else {
					respondError(w, http.StatusInternalServerError, err.Error())
				}
				return
			}
			if err = dstMessageBuilder.Add(event); err != nil {
				if req.Stream {
					slog.Error(fmt.Sprintf("[%d] an error occurs while consuming stream: %s", requestID, err.Error()))
					fmt.Fprintf(w, "event: %s\n", anthropic.EventTypeError)
					if providerError, isProviderError := provider.ParseError(err); isProviderError {
						fmt.Fprintf(w, "data: %s\n\n", utils.JSONEncodeString(&anthropic.StreamError{
							ErrType:    providerError.Type(),
							ErrMessage: providerError.Message(),
						}))
					} else {
						fmt.Fprintf(w, "data: %s\n\n", utils.JSONEncodeString(&anthropic.StreamError{
							ErrType:    anthropic.ErrorContentType,
							ErrMessage: err.Error(),
						}))
					}
				} else {
					if providerError, isProviderError := provider.ParseError(err); isProviderError {
						respondError(w, providerError.StatusCode(), providerError.Message())
					} else {
						respondError(w, http.StatusInternalServerError, err.Error())
					}
				}
				return
			}
			if req.Stream {
				fmt.Fprintf(w, "event: %s\n", event.EventType())
				fmt.Fprintf(w, "data: %s\n\n", utils.JSONEncodeString(event))
				if flusher, isFlusher := w.(http.Flusher); isFlusher {
					flusher.Flush()
				}
			}
			if messageDelta, isMessageDelta := event.(*anthropic.EventMessageDelta); isMessageDelta {
				if messageDelta.Usage != nil {
					inputTokens = messageDelta.Usage.InputTokens
					outputTokens = messageDelta.Usage.OutputTokens
				}
				if messageDelta.Delta != nil && messageDelta.Delta.StopReason != nil {
					stopReason = *messageDelta.Delta.StopReason
				}
			}
		}
		rawBytes, err := json.MarshalIndent(dstMessageBuilder.Message(), "", "    ")
		if err != nil {
			slog.Error(fmt.Sprintf("[%d] error marshaling non-stream response: %s", requestID, err.Error()))
			respondError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if !req.Stream {
			w.WriteHeader(http.StatusOK)
			if _, err = w.Write(rawBytes); err != nil {
				slog.Warn(fmt.Sprintf("[%d] errror sending non-stream response: %s", requestID, err.Error()))
			}
		}
		switch ccProvider {
		case ProviderOpenRouter:
			slog.Info(fmt.Sprintf("[%d] openrouter provider: %s", requestID, orProvider))
		}
		slog.Info(fmt.Sprintf("[%d] stop reason: %s", requestID, stopReason))
		slog.Info(fmt.Sprintf("[%d] final tokens usage: input=%d, output=%d", requestID, inputTokens, outputTokens))
		slog.Debug(fmt.Sprintf(">>>>>>>>>>>>>>>>> [%d] anthropic response >>>>>>>>>>>>>>>>>", requestID) + "\n" + string(rawBytes))
		slog.Debug(fmt.Sprintf("<<<<<<<<<<<<<<<<< [%d] anthropic response <<<<<<<<<<<<<<<<<", requestID))
	}
}

func respondError(w http.ResponseWriter, status int, message string) {
	getSecsToNextMinute := func() int {
		now := time.Now()
		next := now.Add(1 * time.Minute)
		next = time.Date(next.Year(), next.Month(), next.Day(), next.Hour(), next.Minute(), 0, 0, time.Local)
		delta := next.Sub(now)
		return int(delta / time.Second)
	}
	setRetryHeaders := func(secs int) {
		w.Header().Set("Retry-After", strconv.Itoa(secs))
		w.Header().Set("X-Retry-After", strconv.Itoa(secs))
		w.Header().Set("X-Should-Retry", "true")
	}
	var errorType string
	switch status {
	case http.StatusBadRequest:
		errorType = anthropic.InvalidRequestError
	case http.StatusUnauthorized:
		errorType = anthropic.AuthenticationError
	case http.StatusForbidden:
		errorType = anthropic.PermissionError
	case http.StatusNotFound:
		errorType = anthropic.NotFoundError
	case http.StatusRequestEntityTooLarge:
		errorType = anthropic.RequestTooLarge
	case http.StatusTooManyRequests:
		setRetryHeaders(getSecsToNextMinute())
		errorType = anthropic.RateLimitError
	case http.StatusInternalServerError:
		setRetryHeaders(1)
		errorType = anthropic.APIError
	case 529:
		setRetryHeaders(10)
		errorType = anthropic.OverloadedError
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(&anthropic.Error{
		ContentType: anthropic.ErrorContentType,
		Inner: &anthropic.InnerError{
			Type:    errorType,
			Message: message,
		},
	}); err != nil {
		slog.Warn(fmt.Sprintf("[%s] error sending error response: %s", w.Header().Get("X-Cc-Request-Id"), err.Error()))
	}
}
