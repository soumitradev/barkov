package barkov

// NGramSet is a string-keyed anti-verbatim validator. It uses the provided
// StateEncoder to build map keys from n-gram slices, so it works with any
// token type. Zero external dependencies.
type NGramSet[T comparable] struct {
	grams   map[string]struct{}
	n       int
	encoder StateEncoder[T]
}

// NewNGramSet builds an NGramSet from a corpus. n is the n-gram width;
// encoder converts token slices to map keys.
func NewNGramSet[T comparable](corpus [][]T, n int, encoder StateEncoder[T]) *NGramSet[T] {
	grams := make(map[string]struct{})
	for _, tokens := range corpus {
		for i := 0; i <= len(tokens)-n; i++ {
			grams[encoder.Encode(tokens[i:i+n])] = struct{}{}
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
func (s *NGramSet[T]) Validator() func([]T) bool {
	return func(gram []T) bool { return !s.Contains(gram) }
}

// Size returns the number of unique n-grams in the set.
func (s *NGramSet[T]) Size() int { return len(s.grams) }
