package delimiter

import (
	"strings"

	"github.com/spf13/viper"
)

// ViperKeyDelimiter defines the delimiter used to build and parse hierarchical
// configuration keys with Viper across this project.
// Default is "::" so that dots (.) inside YAML map keys are preserved and do
// not get split by Viper.
// All helper calls to ViperKey(...) join segments with this delimiter, and
// init() applies the same delimiter to the global Viper instance via
// viper.SetOptions(viper.KeyDelimiter(...)).
// You can override this value at build time via -ldflags -X, e.g.:
//
//	go build -ldflags="-X 'github.com/x5iu/claude-code-adapter/pkg/utils/delimiter.ViperKeyDelimiter=__'" ./...
//	go test  -ldflags="-X 'github.com/x5iu/claude-code-adapter/pkg/utils/delimiter.ViperKeyDelimiter=__'" ./...
//
// Note: it must remain a package-level string variable (not const) for -X to work.
var ViperKeyDelimiter = "::"

func init() {
	viper.SetOptions(viper.KeyDelimiter(ViperKeyDelimiter))
}

func ViperKey(keys ...string) string {
	return strings.Join(keys, ViperKeyDelimiter)
}
