package barkov

import (
	"slices"
	"testing"
)

func TestSepEncoderRoundTrip(t *testing.T) {
	enc := SepEncoder{Sep: SEP}

	cases := []struct {
		name   string
		tokens []string
	}{
		{"empty", []string{}},
		{"single", []string{"hello"}},
		{"multi", []string{"the", "quick", "brown", "fox"}},
		{"with spaces", []string{"hello world", "foo bar"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := enc.Encode(tc.tokens)
			decoded := enc.Decode(encoded)

			// nil and empty slice should both decode to nil for empty input
			if len(tc.tokens) == 0 {
				if decoded != nil {
					t.Errorf("expected nil for empty input, got %v", decoded)
				}
				return
			}

			if !slices.Equal(decoded, tc.tokens) {
				t.Errorf("round-trip failed: got %v, want %v", decoded, tc.tokens)
			}
		})
	}
}

func TestSepEncoderInjectivity(t *testing.T) {
	enc := SepEncoder{Sep: SEP}

	// Different inputs must produce different outputs
	inputs := [][]string{
		{"a", "b"},
		{"a", "b", "c"},
		{"ab"},
		{"a"},
		{},
	}

	seen := make(map[string][]string)
	for _, tokens := range inputs {
		encoded := enc.Encode(tokens)
		if prev, ok := seen[encoded]; ok {
			t.Errorf("collision: %v and %v both encode to %q", prev, tokens, encoded)
		}
		seen[encoded] = tokens
	}
}
