package delimiter

import (
	"testing"

	"github.com/spf13/viper"
)

func TestViperKey(t *testing.T) {
	tests := []struct {
		name     string
		keys     []string
		expected string
	}{
		{
			name:     "single key",
			keys:     []string{"key1"},
			expected: "key1",
		},
		{
			name:     "two keys",
			keys:     []string{"key1", "key2"},
			expected: "key1::key2",
		},
		{
			name:     "multiple keys",
			keys:     []string{"section", "subsection", "key"},
			expected: "section::subsection::key",
		},
		{
			name:     "empty keys",
			keys:     []string{},
			expected: "",
		},
		{
			name:     "keys with empty strings",
			keys:     []string{"a", "", "b"},
			expected: "a::::b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ViperKey(tt.keys...)
			if result != tt.expected {
				t.Errorf("ViperKey(%v) = %q, want %q", tt.keys, result, tt.expected)
			}
		})
	}
}

func TestViperKeyDelimiterDefault(t *testing.T) {
	expected := "::"
	if ViperKeyDelimiter != expected {
		t.Errorf("ViperKeyDelimiter = %q, want %q", ViperKeyDelimiter, expected)
	}
}

func TestViperIntegration(t *testing.T) {
	// Verify that viper is actually using our delimiter
	// The init() function should have set this up

	// Set a nested value using the delimiter
	key := ViperKey("test", "nested", "value")
	expectedValue := "success"
	viper.Set(key, expectedValue)

	// Verify we can retrieve it using the same key structure
	// If the delimiter wasn't set correctly, viper might treat "test::nested::value" as a single flat key
	// or split it differently.

	// Check if we can access it via the map structure that viper creates
	// When using a custom delimiter, viper creates a nested map structure

	val := viper.GetString(key)
	if val != expectedValue {
		t.Errorf("viper.GetString(%q) = %q, want %q", key, val, expectedValue)
	}

	// Verify structure
	// With "::" delimiter, "test::nested::value" should create a map structure:
	// test:
	//   nested:
	//     value: success

	testMap := viper.GetStringMap("test")
	if testMap == nil {
		t.Fatal("viper.GetStringMap(\"test\") returned nil")
	}

	nestedMap, ok := testMap["nested"].(map[string]interface{})
	if !ok {
		t.Fatalf("test['nested'] is not a map[string]interface{}, got %T", testMap["nested"])
	}

	if nestedMap["value"] != expectedValue {
		t.Errorf("nestedMap['value'] = %v, want %v", nestedMap["value"], expectedValue)
	}
}
