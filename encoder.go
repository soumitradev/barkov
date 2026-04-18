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

// AppendEncoder is an optional optimisation interface. Encoders that
// implement it expose a zero-allocation path for writing the encoded key
// into a caller-supplied byte buffer, which enables arena-style batching
// in hot builders (Chain.Build, NGramSet, nhash). Implementations must
// satisfy the same determinism and injectivity contract as Encode.
type AppendEncoder[T comparable] interface {
	// AppendEncoded appends the encoded form of tokens to dst and returns
	// the extended slice. The appended bytes must equal the bytes of
	// Encode(tokens).
	AppendEncoded(dst []byte, tokens []T) []byte
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

// AppendEncoded writes the joined form of tokens into dst without any
// intermediate allocation. Satisfies AppendEncoder[string].
func (e SepEncoder) AppendEncoded(dst []byte, tokens []string) []byte {
	if len(tokens) == 0 {
		return dst
	}
	dst = append(dst, tokens[0]...)
	for _, t := range tokens[1:] {
		dst = append(dst, e.Sep...)
		dst = append(dst, t...)
	}
	return dst
}

// Decode splits a state string by the separator.
func (e SepEncoder) Decode(state string) []string {
	if state == "" {
		return nil
	}
	return strings.Split(state, e.Sep)
}
