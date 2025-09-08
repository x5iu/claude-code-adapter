package main

import (
	"github.com/spf13/cobra"
)

func main() {
	command := newClaudeClaudeAdapterCliCommand()
	cobra.CheckErr(command.Execute())
}

func newClaudeClaudeAdapterCliCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:           "claude-code-adapter-cli [COMMAND] [OPTIONS]",
		Short:         "Claude Code Adapter Command-Line Interface",
		Version:       "v0.2.0",
		SilenceErrors: true,
		SilenceUsage:  true,
	}
	cmd.AddCommand(newServeCommand())
	return cmd
}
