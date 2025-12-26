package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/samber/lo"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/x5iu/claude-code-adapter/pkg/adapter"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/anthropic"
	"github.com/x5iu/claude-code-adapter/pkg/datatypes/openrouter"
	"github.com/x5iu/claude-code-adapter/pkg/profile"
	"github.com/x5iu/claude-code-adapter/pkg/provider"
	"github.com/x5iu/claude-code-adapter/pkg/snapshot"
	"github.com/x5iu/claude-code-adapter/pkg/snapshot/jsonl"
	"github.com/x5iu/claude-code-adapter/pkg/utils"
	"github.com/x5iu/claude-code-adapter/pkg/utils/delimiter"
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
			if viper.GetBool(delimiter.ViperKey("debug")) {
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
	flags.String("host", "127.0.0.1", "host to serve on")
	flags.String("snapshot", "", "snapshot recorder config")
	cobra.CheckErr(viper.BindPFlag(delimiter.ViperKey("debug"), flags.Lookup("debug")))
	cobra.CheckErr(viper.BindPFlag(delimiter.ViperKey("http", "port"), flags.Lookup("port")))
	cobra.CheckErr(viper.BindPFlag(delimiter.ViperKey("http", "host"), flags.Lookup("host")))
	cobra.CheckErr(viper.BindPFlag(delimiter.ViperKey("snapshot"), flags.Lookup("snapshot")))
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
	// Load profiles from configuration
	var profileManagerPtr atomic.Pointer[profile.ProfileManager]
	loadProfiles := func() error {
		pm, err := profile.LoadFromViper(viper.GetViper())
		if err != nil {
			return err
		}
		profileManagerPtr.Store(pm)
		slog.Info(fmt.Sprintf("loaded %d profiles", len(pm.Profiles())))
		for _, p := range pm.Profiles() {
			slog.Debug(fmt.Sprintf("profile %q: provider=%s, models=%v", p.Name, p.Provider, p.Models))
		}
		return nil
	}
	if err := loadProfiles(); err != nil {
		cobra.CheckErr(fmt.Errorf("profile: %w", err))
	}
	viper.OnConfigChange(func(fsnotify.Event) {
		slog.Info("config file changed, reloading")
		if err := loadProfiles(); err != nil {
			slog.Error(fmt.Sprintf("error reloading profiles: %s", err.Error()))
		}
	})
	viper.WatchConfig()
	recorder, err := makeSnapshotRecorder(ctx, viper.GetString(delimiter.ViperKey("snapshot")))
	if err != nil {
		cobra.CheckErr(fmt.Errorf("snapshot: %w", err))
	}
	defer recorder.Close()
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/messages", onMessages(cmd, provider.NewProvider(), recorder, &profileManagerPtr))
	mux.HandleFunc("/v1/messages/count_tokens", onCountTokens(&profileManagerPtr))
	server := &http.Server{
		Addr:     fmt.Sprintf("%s:%d", viper.GetString(delimiter.ViperKey("http", "host")), viper.GetUint16(delimiter.ViperKey("http", "port"))),
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

func onMessages(cmd *cobra.Command, prov provider.Provider, rec snapshot.Recorder, pmPtr *atomic.Pointer[profile.ProfileManager]) func(w http.ResponseWriter, r *http.Request) {
	var (
		requestCounter atomic.Int64
		version        = cmd.Parent().Version
	)
	return func(w http.ResponseWriter, r *http.Request) {
		var matchedProfileConfig *snapshot.Config
		sn := &snapshot.Snapshot{
			RequestTime: time.Now(),
			Version:     version,
		}
		requestID := requestCounter.Add(1)
		sn.RequestID = strconv.FormatInt(requestID, 10)
		defer func() {
			go func() {
				sn.FinishTime = time.Now()
				sn.RequestHeader = snapshot.Header(r.Header)
				// record matched profile config instead of global config
				sn.Config = matchedProfileConfig
				if err := rec.Record(sn); err != nil {
					slog.Warn(fmt.Sprintf("[%d] error recording snapshot: %s", requestID, err.Error()))
				}
			}()
		}()
		removeForwardedHeaders(r.Header)
		r.Header.Del(anthropic.HeaderAPIKey)
		r.Header.Set("User-Agent", fmt.Sprintf("claude-code-adapter-cli/%s", version[1:]))
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
				sn.StatusCode = http.StatusInternalServerError
			}
		}()
		if !utils.IsContentType(r.Header, "application/json") {
			respondError(w,
				http.StatusBadRequest,
				fmt.Sprintf("Invalid Content-Type %q", r.Header.Get("Content-Type")),
			)
			sn.StatusCode = http.StatusBadRequest
			return
		}
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			respondError(w,
				http.StatusInternalServerError,
				fmt.Sprintf("Failed to read request body: %s", err.Error()),
			)
			sn.Error = &snapshot.Error{Message: err.Error()}
			sn.StatusCode = http.StatusInternalServerError
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
			sn.Error = &snapshot.Error{Message: err.Error()}
			sn.StatusCode = http.StatusBadRequest
			return
		}
		sn.AnthropicRequest = req
		r.Header.Del("Content-Type")
		r.Header.Del("Content-Length")
		r.Header.Del("Transfer-Encoding")
		r.Header.Del("Accept-Encoding")
		slog.Info(fmt.Sprintf("[%d] request model: %s", requestID, req.Model))
		// Match profile for the requested model
		prof, err := pmPtr.Load().Match(req.Model)
		if err != nil {
			slog.Error(fmt.Sprintf("[%d] no profile matched for model %q: %s", requestID, req.Model, err.Error()))
			respondError(w, http.StatusBadRequest, fmt.Sprintf("No profile configured for model %q", req.Model))
			sn.Error = &snapshot.Error{Message: err.Error()}
			sn.StatusCode = http.StatusBadRequest
			return
		}
		slog.Info(fmt.Sprintf("[%d] matched profile: %s (provider=%s)", requestID, prof.Name, prof.Provider))
		sn.Profile = prof.Name
		matchedProfileConfig = profileToSnapshotConfig(prof)
		// Inject profile into request context
		ctx := profile.WithProfile(r.Context(), prof)
		// Remove disallowed tools as early as possible (ingress filtering)
		if len(req.Tools) > 0 {
			// Build a disallowed tool name set from profile options
			disallowedSet := map[string]struct{}{}
			for _, name := range prof.Options.GetDisallowedTools() {
				if name == "" {
					continue
				}
				disallowedSet[name] = struct{}{}
			}
			if len(disallowedSet) > 0 {
				filtered := make([]*anthropic.Tool, 0, len(req.Tools))
				removed := make([]string, 0)
				for _, t := range req.Tools {
					if _, blocked := disallowedSet[t.Name]; blocked {
						removed = append(removed, t.Name)
						continue
					}
					filtered = append(filtered, t)
				}
				if len(removed) > 0 {
					slog.Info(fmt.Sprintf("[%d] removed disallowed tools: %s", requestID, strings.Join(removed, ",")))
				}
				req.Tools = filtered
				// Normalize tool_choice if necessary
				if len(req.Tools) == 0 {
					if req.ToolChoice == nil {
						req.ToolChoice = &anthropic.ToolChoice{Type: anthropic.ToolChoiceTypeNone}
					} else {
						req.ToolChoice.Type = anthropic.ToolChoiceTypeNone
						req.ToolChoice.Name = ""
					}
				} else if req.ToolChoice != nil && req.ToolChoice.Type == anthropic.ToolChoiceTypeTool {
					// Ensure selected tool still exists after filtering
					remaining := map[string]struct{}{}
					for _, t := range req.Tools {
						remaining[t.Name] = struct{}{}
					}
					if _, ok := remaining[req.ToolChoice.Name]; !ok {
						req.ToolChoice.Type = anthropic.ToolChoiceTypeNone
						req.ToolChoice.Name = ""
					}
				}
			}
		}
		if prof.Options.GetPreventEmptyTextToolResult() {
			// No idea why Claude Code send empty text in tool_result, so we replace it with a hint message if necessary.
			for _, message := range req.Messages {
				if message != nil {
					for _, content := range message.Content {
						if content != nil && content.Type == anthropic.MessageContentTypeToolResult {
							for _, part := range content.Content {
								if part != nil && part.Type == anthropic.MessageContentTypeText && part.Text == "" {
									part.Text = "(No content)"
								}
							}
						}
					}
				}
			}
		}
		var (
			inputTokens  int64
			outputTokens int64
			stopReason   = anthropic.StopReason("unknown")
		)
		countTokensCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		// When tool.type is null, the /v1/messages/count_tokens endpoint will return the error
		// tools.0.get_defaulted_tool_discriminator(): Field required, so we need to preprocess tools to avoid this error.
		for _, tool := range req.Tools {
			if tool.Type == nil {
				tool.Type = lo.ToPtr(anthropic.ToolTypeCustom)
			}
			if prof.Anthropic.GetDisableWebSearchBlockedDomains() {
				if *tool.Type == anthropic.ToolTypeCustom && tool.Name == anthropic.ToolNameWebSearch {
					if schemaType := gjson.GetBytes(tool.InputSchema, "type"); schemaType.String() == "object" {
						key := fmt.Sprintf("properties.%s", "blocked_domains")
						newInputSchema, err := sjson.DeleteBytes(tool.InputSchema, key)
						if err == nil {
							tool.InputSchema = newInputSchema
						} else {
							slog.Warn(fmt.Sprintf("[%d] error disabling %q in %s tool: %s", requestID, key, anthropic.ToolNameWebSearch, err.Error()))
						}
					}
				}
			}
		}
		if !prof.Options.GetDisableCountTokensRequest() {
			usage, err := prov.CountAnthropicTokens(countTokensCtx, &anthropic.CountTokensRequest{
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
				sn.Error = &snapshot.Error{Message: err.Error()}
			} else {
				inputTokens = usage.InputTokens
				slog.Info(fmt.Sprintf("[%d] request input tokens (estimated): %d", requestID, inputTokens))
			}
		}
		hasServerTools := sync.OnceValue(func() bool {
			return lo.ContainsBy(req.Tools, func(tool *anthropic.Tool) bool {
				isServerTool := tool.Type != nil && *tool.Type != anthropic.ToolTypeCustom
				if isServerTool {
					slog.Info(fmt.Sprintf("[%d] request contains %s tool: %s", requestID, *tool.Type, tool.Name))
				}
				return isServerTool
			})
		})
		useAnthropicProvider := func() bool {
			return hasServerTools() || prof.Provider == ProviderAnthropic
		}
		var (
			stream     anthropic.MessageStream
			ccProvider = prof.Provider
			orProvider = "<unknown>"
		)
		defer func() {
			if stream != nil {
				for range stream {
				}
			}
		}()
		if useAnthropicProvider() {
			sn.Provider = ProviderAnthropic
			slog.Info(fmt.Sprintf("[%d] using provider %q", requestID, ProviderAnthropic))
			w.Header().Set("X-Provider", ProviderAnthropic)
			var (
				header                http.Header
				reader                io.ReadCloser
				enablePassThroughMode = prof.Anthropic.GetEnablePassThroughMode()
			)
			defer func() {
				sn.ResponseHeader = snapshot.Header(header)
			}()
			if enablePassThroughMode {
				reader, header, err = prov.MakeAnthropicMessagesRequest(ctx,
					utils.NewResettableReader(rawBody),
					provider.WithQuery("beta", "true"),
					provider.WithHeaders(r.Header),
				)
			} else {
				options := []provider.RequestOption{
					provider.WithQuery("beta", "true"),
					provider.WithHeaders(r.Header),
				}
				if prof.Anthropic.GetUseRawRequestBody() {
					rawBody, err = sjson.SetBytes(rawBody, "stream", true)
					if err != nil {
						panic(fmt.Errorf("unreachable: %s", err.Error()))
					}
					options = append(options, provider.ReplaceBody(rawBody))
				}
				stream, header, err = prov.GenerateAnthropicMessage(ctx, req, options...)
			}
			if err != nil {
				slog.Error(fmt.Sprintf("[%d] error making anthropic /v1/messages request: %s", requestID, err.Error()))
				if providerError, isProviderError := provider.ParseError(err); isProviderError {
					respondError(w, providerError.StatusCode(), providerError.Message())
					sn.Error = &snapshot.Error{
						Message: providerError.Message(),
						Type:    providerError.Type(),
						Source:  providerError.Source(),
					}
					sn.StatusCode = providerError.StatusCode()
				} else {
					respondError(w, http.StatusInternalServerError, err.Error())
					sn.Error = &snapshot.Error{Message: err.Error()}
					sn.StatusCode = http.StatusInternalServerError
				}
				return
			}
			if enablePassThroughMode {
				defer reader.Close()
				for k, vv := range header {
					for _, v := range vv {
						w.Header().Add(k, v)
					}
				}
				w.Header().Del("Content-Length")
				w.Header().Del("Content-Encoding")
				w.WriteHeader(http.StatusOK)
				sn.StatusCode = http.StatusOK
				// Copy response to client and capture for parsing
				var recvBuf bytes.Buffer
				tee := io.TeeReader(reader, &recvBuf)
				if _, err := io.Copy(w, tee); err != nil {
					slog.Warn(fmt.Sprintf("[%d] error sending Anthropic response: %s", requestID, err))
					sn.Error = &snapshot.Error{Message: err.Error()}
				}
				// Parse response to build anthropic.Message for snapshot
				// Check Content-Type header to determine format
				if utils.IsContentType(header, "text/event-stream") {
					// SSE format: use MakeAnthropicStream and MessageBuilder
					dstMessageBuilder := anthropic.NewMessageBuilder()
					stream := provider.MakeAnthropicStream(prof, io.NopCloser(&recvBuf))
					for event, err := range stream {
						if err != nil {
							slog.Error(fmt.Sprintf("[%d] error parsing SSE response for snapshot: %s", requestID, err))
							sn.Error = &snapshot.Error{Message: err.Error()}
							return
						}
						if err = dstMessageBuilder.Add(event); err != nil {
							slog.Error(fmt.Sprintf("[%d] error building message: %s", requestID, err))
							sn.Error = &snapshot.Error{Message: err.Error()}
							return
						}
					}
					sn.AnthropicResponse = dstMessageBuilder.Message()
				} else if utils.IsContentType(header, "application/json") {
					// JSON format: unmarshal directly
					if err := json.Unmarshal(recvBuf.Bytes(), &sn.AnthropicResponse); err != nil {
						slog.Error(fmt.Sprintf("[%d] error unmarshalling Anthropic response: %s", requestID, err))
						sn.Error = &snapshot.Error{Message: err.Error()}
					}
				} else {
					slog.Warn(fmt.Sprintf("[%d] unknown Content-Type for snapshot parsing: %s", requestID, header.Get("Content-Type")))
				}
				// Log response for debugging (only after parsing to avoid affecting recvBuf)
				slog.Debug(fmt.Sprintf(">>>>>>>>>>>>>>>>> [%d] anthropic response >>>>>>>>>>>>>>>>>", requestID) + "\n" + recvBuf.String())
				slog.Debug(fmt.Sprintf("<<<<<<<<<<<<<<<<< [%d] anthropic response <<<<<<<<<<<<<<<<<", requestID))
				return
			}
		} else {
			switch ccProvider {
			case ProviderOpenRouter:
				fallthrough
			default:
				sn.Provider = ProviderOpenRouter
				ccProvider = ProviderOpenRouter
				slog.Info(fmt.Sprintf("[%d] using provider %q", requestID, ProviderOpenRouter))
				w.Header().Set("X-Provider", ProviderOpenRouter)
				allowedProviders := prof.OpenRouter.GetAllowedProviders()
				openrouterRequest := adapter.ConvertAnthropicRequestToOpenRouterRequest(ctx, req)
				sn.OpenRouterRequest = openrouterRequest
				orStream, header, err := prov.CreateOpenRouterChatCompletion(
					ctx,
					openrouterRequest,
					openrouter.WithIdentity("https://github.com/x5iu/claude-code-adapter", "claude-code-adapter"),
					openrouter.WithAnthropicBetaFeatures(r.Header),
					openrouter.WithProviderPreference(&openrouter.ProviderPreference{
						Order:             allowedProviders,
						AllowFallbacks:    lo.ToPtr(true),
						RequireParameters: lo.ToPtr(false), // OpenRouter does not support all Anthropic parameters.
						Only:              allowedProviders,
						Sort:              lo.ToPtr(openrouter.ProviderSortMethodThroughput),
					}),
				)
				chatCompletionBuilder := openrouter.NewChatCompletionBuilder()
				defer func() {
					sn.ResponseHeader = snapshot.Header(header)
					sn.OpenRouterResponse = chatCompletionBuilder.Build()
				}()
				if err != nil {
					slog.Error(fmt.Sprintf("[%d] error making OpenRouter ChatCompletions request: %s", requestID, err.Error()))
					if providerError, isProviderError := provider.ParseError(err); isProviderError {
						respondError(w, providerError.StatusCode(), providerError.Message())
						sn.Error = &snapshot.Error{
							Message: providerError.Message(),
							Type:    providerError.Type(),
							Source:  providerError.Source(),
						}
						sn.StatusCode = providerError.StatusCode()
					} else {
						respondError(w, http.StatusInternalServerError, err.Error())
						sn.Error = &snapshot.Error{Message: err.Error()}
						sn.StatusCode = http.StatusInternalServerError
					}
					return
				}
				stream = adapter.ConvertOpenRouterStreamToAnthropicStream(
					ctx,
					orStream,
					adapter.WithInputTokens(inputTokens),
					adapter.ExtractOpenRouterProvider(&orProvider),
					adapter.ExtractOpenRouterChatCompletionBuilder(chatCompletionBuilder),
				)
			}
		}
		dstMessageBuilder := anthropic.NewMessageBuilder()
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			sn.StatusCode = http.StatusOK
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		contextWindowResizeFactor := prof.Options.GetContextWindowResizeFactor()
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
					sn.Error = &snapshot.Error{Message: err.Error()}
					sn.StatusCode = http.StatusInternalServerError
				}
				return
			}
			switch e := event.(type) {
			case *anthropic.EventMessageStart:
				if e.Message != nil {
					if usage := e.Message.Usage; usage != nil {
						usage.InputTokens = int64(float64(usage.InputTokens) * contextWindowResizeFactor)
						usage.OutputTokens = int64(float64(usage.OutputTokens) * contextWindowResizeFactor)
						usage.CacheReadInputTokens = int64(float64(usage.CacheReadInputTokens) * contextWindowResizeFactor)
						usage.CacheCreationInputTokens = int64(float64(usage.CacheCreationInputTokens) * contextWindowResizeFactor)
						if cacheCreation := usage.CacheCreation; cacheCreation != nil {
							cacheCreation.Ephemeral5MInputTokens = int64(float64(cacheCreation.Ephemeral5MInputTokens) * contextWindowResizeFactor)
							cacheCreation.Ephemeral1HInputTokens = int64(float64(cacheCreation.Ephemeral1HInputTokens) * contextWindowResizeFactor)
						}
					}
				}
			case *anthropic.EventMessageDelta:
				if e.Usage != nil {
					e.Usage.InputTokens = int64(float64(e.Usage.InputTokens) * contextWindowResizeFactor)
					e.Usage.OutputTokens = int64(float64(e.Usage.OutputTokens) * contextWindowResizeFactor)
					e.Usage.CacheReadInputTokens = int64(float64(e.Usage.CacheReadInputTokens) * contextWindowResizeFactor)
					e.Usage.CacheCreationInputTokens = int64(float64(e.Usage.CacheCreationInputTokens) * contextWindowResizeFactor)
					if cacheCreation := e.Usage.CacheCreation; cacheCreation != nil {
						cacheCreation.Ephemeral5MInputTokens = int64(float64(cacheCreation.Ephemeral5MInputTokens) * contextWindowResizeFactor)
						cacheCreation.Ephemeral1HInputTokens = int64(float64(cacheCreation.Ephemeral1HInputTokens) * contextWindowResizeFactor)
					}
				}
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
						sn.Error = &snapshot.Error{
							Message: providerError.Message(),
							Type:    providerError.Type(),
							Source:  providerError.Source(),
						}
					} else {
						fmt.Fprintf(w, "data: %s\n\n", utils.JSONEncodeString(&anthropic.StreamError{
							ErrType:    anthropic.ErrorContentType,
							ErrMessage: err.Error(),
						}))
						sn.Error = &snapshot.Error{Message: err.Error()}
					}
				} else {
					if providerError, isProviderError := provider.ParseError(err); isProviderError {
						respondError(w, providerError.StatusCode(), providerError.Message())
						sn.Error = &snapshot.Error{
							Message: providerError.Message(),
							Type:    providerError.Type(),
							Source:  providerError.Source(),
						}
						sn.StatusCode = providerError.StatusCode()
					} else {
						respondError(w, http.StatusInternalServerError, err.Error())
						sn.Error = &snapshot.Error{Message: err.Error()}
						sn.StatusCode = http.StatusInternalServerError
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
			if messageStart, isMessageStart := event.(*anthropic.EventMessageStart); isMessageStart {
				if messageStart.Message != nil {
					if usage := messageStart.Message.Usage; usage != nil {
						inputTokens = usage.InputTokens
					}
				}
			}
			if messageDelta, isMessageDelta := event.(*anthropic.EventMessageDelta); isMessageDelta {
				if messageDelta.Usage != nil {
					if messageDelta.Usage.InputTokens > 0 {
						inputTokens = messageDelta.Usage.InputTokens
					}
					outputTokens = messageDelta.Usage.OutputTokens
				}
				if messageDelta.Delta != nil && messageDelta.Delta.StopReason != nil {
					stopReason = *messageDelta.Delta.StopReason
				}
			}
		}
		dstMessage := dstMessageBuilder.Message()
		sn.AnthropicResponse = dstMessage
		rawBytes, err := json.MarshalIndent(dstMessage, "", "    ")
		if err != nil {
			slog.Error(fmt.Sprintf("[%d] error marshaling non-stream response: %s", requestID, err.Error()))
			respondError(w, http.StatusInternalServerError, err.Error())
			sn.Error = &snapshot.Error{Message: err.Error()}
			sn.StatusCode = http.StatusInternalServerError
			return
		}
		if !req.Stream {
			w.WriteHeader(http.StatusOK)
			sn.StatusCode = http.StatusOK
			if _, err = w.Write(rawBytes); err != nil {
				slog.Warn(fmt.Sprintf("[%d] errror sending non-stream response: %s", requestID, err.Error()))
				sn.Error = &snapshot.Error{Message: err.Error()}
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

func profileToSnapshotConfig(p *profile.Profile) *snapshot.Config {
	if p == nil {
		return nil
	}
	cfg := &snapshot.Config{Provider: p.Provider}
	if p.Options != nil {
		cfg.Options = &snapshot.OptionsConfig{
			Strict:                     p.Options.Strict,
			PreventEmptyTextToolResult: p.Options.PreventEmptyTextToolResult,
			Models:                     p.Options.Models,
			ContextWindowResizeFactor:  p.Options.ContextWindowResizeFactor,
			DisableCountTokensRequest:  p.Options.DisableCountTokensRequest,
			MinMaxTokens:               p.Options.MinMaxTokens,
		}
		if p.Options.Reasoning != nil {
			cfg.Options.Reasoning = &snapshot.ReasoningConfig{
				Format:    p.Options.Reasoning.Format,
				Effort:    p.Options.Reasoning.Effort,
				Delimiter: p.Options.Reasoning.Delimiter,
			}
		}
	}
	if p.Anthropic != nil {
		cfg.Anthropic = &snapshot.AnthropicConfig{
			UseRawRequestBody:              p.Anthropic.UseRawRequestBody,
			EnablePassThroughMode:          p.Anthropic.EnablePassThroughMode,
			DisableWebSearchBlockedDomains: p.Anthropic.DisableWebSearchBlockedDomains,
			ForceThinking:                  p.Anthropic.ForceThinking,
			BaseURL:                        p.Anthropic.BaseURL,
			Version:                        p.Anthropic.Version,
		}
	}
	if p.OpenRouter != nil {
		cfg.OpenRouter = &snapshot.OpenRouterConfig{
			BaseURL:              p.OpenRouter.BaseURL,
			ModelReasoningFormat: p.OpenRouter.ModelReasoningFormat,
			AllowedProviders:     p.OpenRouter.AllowedProviders,
		}
	}
	return cfg
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

func makeSnapshotRecorder(ctx context.Context, cfg string) (snapshot.Recorder, error) {
	if cfg == "" {
		return snapshot.NopRecorder(), nil
	}
	u, err := url.Parse(cfg)
	if err != nil {
		return nil, err
	}
	switch u.Scheme {
	case "jsonl":
		var path string
		if u.Opaque != "" {
			path = u.Opaque
		} else {
			path = u.Path
		}
		file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, err
		}
		return jsonl.NewRecorder(ctx, file), nil
	default:
		return nil, fmt.Errorf("unsupported snapshot recorder type %q", u.Scheme)
	}
}

func onCountTokens(pmPtr *atomic.Pointer[profile.ProfileManager]) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		removeForwardedHeaders(r.Header)
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
		model := gjson.GetBytes(rawBody, "model").String()
		if model == "" {
			respondError(w, http.StatusBadRequest, "Missing required field: model")
			return
		}
		prof, err := pmPtr.Load().Match(model)
		if err != nil {
			respondError(w, http.StatusBadRequest, fmt.Sprintf("No profile configured for model %q", model))
			return
		}
		countTokensBackend := prof.Anthropic.GetCountTokensBackend()
		backendURL, err := url.Parse(countTokensBackend)
		if err != nil {
			respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to parse count tokens backend URL: %s", err.Error()))
			slog.Error(fmt.Sprintf("failed to parse count tokens backend URL %s: %s", countTokensBackend, err.Error()))
			return
		}
		r.Host = backendURL.Host
		r.Body = io.NopCloser(bytes.NewReader(rawBody))
		r.ContentLength = int64(len(rawBody))
		r.Header.Set("Host", backendURL.Host)
		r.Header.Set("Content-Length", strconv.Itoa(len(rawBody)))
		r.Header.Set(anthropic.HeaderAPIKey, prof.Anthropic.GetAPIKey())
		proxy := httputil.NewSingleHostReverseProxy(backendURL)
		proxy.ServeHTTP(w, r)
	}
}

func removeForwardedHeaders(header http.Header) {
	header.Del("Forwarded")
	header.Del("X-Forwarded-For")
	header.Del("X-Forwarded-Host")
	header.Del("X-Forwarded-Port")
	header.Del("X-Forwarded-Proto")
	header.Del("X-Forwarded-Scheme")
}
