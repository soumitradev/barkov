package barkov

import "strings"

// StateEncoder turns a token window into a map key and back. Encode must be
// deterministic and injective, and Decode(Encode(x)) must equal x.
type StateEncoder[T comparable] interface {
	Encode(tokens []T) string
	Decode(state string) []T
}

// AppendEncoder lets builders encode keys without allocating. AppendEncoded
// must append exactly the bytes of Encode(tokens) to dst.
type AppendEncoder[T comparable] interface {
	AppendEncoded(dst []byte, tokens []T) []byte
}

// SepEncoder joins string tokens with Sep, which must not appear in any token.
type SepEncoder struct {
	Sep string
}

// DefaultStringEncoder matches the encoder InitChain gives string chains.
var DefaultStringEncoder = SepEncoder{Sep: SEP}

// Encode joins tokens with the separator.
func (e SepEncoder) Encode(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}
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

// AppendEncoded appends the joined tokens to dst.
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
