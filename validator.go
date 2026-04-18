package barkov

import "unsafe"

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

	if appendEnc, ok := any(encoder).(AppendEncoder[T]); ok {
		arena := newKeyArena(total/4*n*8 + 64)
		scratch := make([]byte, 0, 256)
		for _, tokens := range corpus {
			for i := 0; i <= len(tokens)-n; i++ {
				scratch = scratch[:0]
				scratch = appendEnc.AppendEncoded(scratch, tokens[i:i+n])
				probe := unsafe.String(unsafe.SliceData(scratch), len(scratch))
				if _, ok := grams[probe]; ok {
					continue
				}
				grams[arena.Append(scratch)] = struct{}{}
			}
		}
	} else {
		for _, tokens := range corpus {
			for i := 0; i <= len(tokens)-n; i++ {
				grams[encoder.Encode(tokens[i:i+n])] = struct{}{}
			}
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
