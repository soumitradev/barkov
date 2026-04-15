package barkov

import "strings"

// StateEncoder converts a sequence of tokens into an opaque string key used
// for map lookups, and back. Implementations must be:
//  1. Deterministic: same input → same output, always.
//  2. Injective: different inputs → different outputs.
//  3. Round-trippable: Decode(Encode(x)) must equal x.
//
// The returned key from Encode is used as a Go map key, so it must be safe
// to store long-term and cheap to compare. Go strings are ideal because
// they're immutable and have optimized hashing.
type StateEncoder[T comparable] interface {
	Encode(tokens []T) string
	Decode(state string) []T
}

// SepEncoder joins strings with a separator. The separator must not appear
// in any real token. Used as the default for string chains.
type SepEncoder struct {
	Sep string
}

// Encode joins tokens with the separator.
func (e SepEncoder) Encode(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}
	// Pre-calculate total length for efficiency
	n := len(e.Sep) * (len(tokens) - 1)
	for _, t := range tokens {
		n += len(t)
	}
	var sb strings.Builder
	sb.Grow(n)
	sb.WriteString(tokens[0])
	for _, t := range tokens[1:] {
		sb.WriteString(e.Sep)
		sb.WriteString(t)
	}
	return sb.String()
}

// Decode splits a state string by the separator.
func (e SepEncoder) Decode(state string) []string {
	if state == "" {
		return nil
	}
	return strings.Split(state, e.Sep)
}
