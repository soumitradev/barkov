package barkov

import "unsafe"

// WindowValidator is a per-window validator that knows its own width.
// Gen slides a window of exactly N() tokens, eliminating the silent
// width mismatch that WithValidator has with hand-wired NGramSets.
type WindowValidator[T comparable] interface {
	Validate(gram []T) bool
	N() int
}

// NGramSet is a string-keyed anti-verbatim validator. It uses the provided
// StateEncoder to build map keys from n-gram slices, so it works with any
// token type. Zero external dependencies.
type NGramSet[T comparable] struct {
	grams   map[string]struct{}
	n       int
	encoder StateEncoder[T]
}

// NewNGramSet builds an NGramSet from a corpus. n is the n-gram width;
// encoder converts token slices to map keys. When the encoder implements
// AppendEncoder[T] the build skips per-gram string allocation by probing
// the map with a scratch-backed view, only interning bytes into an arena
// on the first occurrence of each unique n-gram.
func NewNGramSet[T comparable](corpus [][]T, n int, encoder StateEncoder[T]) *NGramSet[T] {
	total := 0
	for _, tokens := range corpus {
		if len(tokens) >= n {
			total += len(tokens) - n + 1
		}
	}
	grams := make(map[string]struct{}, total/2+16)
	ks := newKeyStream(encoder, total/4*n*8+64)

	for _, tokens := range corpus {
		for i := 0; i <= len(tokens)-n; i++ {
			probeKey := ks.probe(tokens[i : i+n])
			if _, seen := grams[probeKey]; seen {
				continue
			}
			grams[ks.intern(probeKey)] = struct{}{}
		}
	}
	return &NGramSet[T]{grams: grams, n: n, encoder: encoder}
}

// Contains reports whether gram is in the set.
func (s *NGramSet[T]) Contains(gram []T) bool {
	_, ok := s.grams[s.encoder.Encode(gram)]
	return ok
}

// Validator returns a function suitable for WithValidator that rejects any
// gram present in the set (anti-verbatim: blocks exact corpus n-grams).
// When the encoder implements AppendEncoder[T], the returned closure uses
// a stack-backed scratch buffer to probe the map without allocation.
func (s *NGramSet[T]) Validator() func([]T) bool {
	grams := s.grams
	if appendEnc, ok := any(s.encoder).(AppendEncoder[T]); ok {
		return func(gram []T) bool {
			var buf [256]byte
			scratch := appendEnc.AppendEncoded(buf[:0], gram)
			probe := unsafe.String(unsafe.SliceData(scratch), len(scratch))
			_, found := grams[probe]
			return !found
		}
	}
	encoder := s.encoder
	return func(gram []T) bool {
		_, found := grams[encoder.Encode(gram)]
		return !found
	}
}

// Size returns the number of unique n-grams in the set.
func (s *NGramSet[T]) Size() int { return len(s.grams) }

// Validate implements WindowValidator[T]. Same accept-semantics as the
// Validator() closure: true means gram is NOT a verbatim corpus n-gram.
// Mirrors Validator()'s allocation-free append-encoder fast path rather
// than allocating a fresh closure on every call.
func (s *NGramSet[T]) Validate(gram []T) bool {
	if appendEnc, ok := any(s.encoder).(AppendEncoder[T]); ok {
		var buf [256]byte
		scratch := appendEnc.AppendEncoded(buf[:0], gram)
		probe := unsafe.String(unsafe.SliceData(scratch), len(scratch))
		_, found := s.grams[probe]
		return !found
	}
	_, found := s.grams[s.encoder.Encode(gram)]
	return !found
}

// N implements WindowValidator[T], returning the n-gram width this set
// was built with.
func (s *NGramSet[T]) N() int { return s.n }

var _ WindowValidator[string] = (*NGramSet[string])(nil)
