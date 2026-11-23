package barkov

import (
	"sync"
	"unsafe"
)

const maxPooledScratch = 4096

var scratchPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 256)
		return &b
	},
}

// WindowValidator is a per-window validator that declares its own width so
// Gen slides exactly N() tokens.
type WindowValidator[T comparable] interface {
	Validate(gram []T) bool
	N() int
}

// NGramSet is a string-keyed anti-verbatim validator.
type NGramSet[T comparable] struct {
	grams     map[string]struct{}
	n         int
	encoder   StateEncoder[T]
	appendEnc AppendEncoder[T]
}

// NewNGramSet builds an NGramSet of n-grams of width n from corpus.
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
	return &NGramSet[T]{grams: grams, n: n, encoder: encoder, appendEnc: ks.appendEnc}
}

// Contains reports whether gram is in the set.
func (s *NGramSet[T]) Contains(gram []T) bool {
	if s.appendEnc != nil {
		bp := scratchPool.Get().(*[]byte)
		buf := s.appendEnc.AppendEncoded(*bp, gram)
		// map lookup does not retain the key, so buf can return to the pool right after
		_, ok := s.grams[unsafe.String(unsafe.SliceData(buf), len(buf))]
		if cap(buf) <= maxPooledScratch {
			*bp = buf[:0]
			scratchPool.Put(bp)
		}
		return ok
	}
	_, ok := s.grams[s.encoder.Encode(gram)]
	return ok
}

// Validate implements WindowValidator[T], rejecting corpus n-grams.
func (s *NGramSet[T]) Validate(gram []T) bool { return !s.Contains(gram) }

// Validator returns a WithValidator callback that rejects corpus n-grams.
func (s *NGramSet[T]) Validator() func([]T) bool { return s.Validate }

// Size returns the number of unique n-grams in the set.
func (s *NGramSet[T]) Size() int { return len(s.grams) }

// N implements WindowValidator[T], returning the n-gram width.
func (s *NGramSet[T]) N() int { return s.n }

var _ WindowValidator[string] = (*NGramSet[string])(nil)
