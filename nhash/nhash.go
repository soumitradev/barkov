// Package nhash provides a hash-keyed anti-verbatim validator that stores
// uint64 hashes instead of full n-gram strings.
package nhash

import (
	"sync"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers"
)

const maxPooledScratch = 4096

var scratchPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 256)
		return &b
	},
}

// HashNGramSet is a hash-keyed anti-verbatim validator.
type HashNGramSet[T comparable] struct {
	hashes    map[uint64]struct{}
	n         int
	hasher    hashers.Hasher
	encoder   barkov.StateEncoder[T]
	appendEnc barkov.AppendEncoder[T]
}

// New builds a HashNGramSet of n-grams of width n from corpus.
func New[T comparable](
	corpus [][]T,
	n int,
	encoder barkov.StateEncoder[T],
	hasher hashers.Hasher,
) *HashNGramSet[T] {
	total := 0
	for _, tokens := range corpus {
		if len(tokens) >= n {
			total += len(tokens) - n + 1
		}
	}
	set := make(map[uint64]struct{}, total/2+16)
	appendEnc, _ := any(encoder).(barkov.AppendEncoder[T])

	var scratch []byte
	if appendEnc != nil {
		scratch = make([]byte, 0, 256)
	}
	for _, tokens := range corpus {
		for i := 0; i <= len(tokens)-n; i++ {
			gram := tokens[i : i+n]
			var h uint64
			if appendEnc != nil {
				scratch = appendEnc.AppendEncoded(scratch[:0], gram)
				h = hasher.Hash(scratch)
			} else {
				h = hashString(hasher, encoder.Encode(gram))
			}
			set[h] = struct{}{}
		}
	}
	return &HashNGramSet[T]{
		hashes:    set,
		n:         n,
		hasher:    hasher,
		encoder:   encoder,
		appendEnc: appendEnc,
	}
}

// Contains reports whether gram is in the set.
func (s *HashNGramSet[T]) Contains(gram []T) bool {
	var h uint64
	if s.appendEnc != nil {
		bp := scratchPool.Get().(*[]byte)
		buf := s.appendEnc.AppendEncoded(*bp, gram)
		h = s.hasher.Hash(buf)
		if cap(buf) <= maxPooledScratch {
			*bp = buf[:0]
			scratchPool.Put(bp)
		}
	} else {
		h = hashString(s.hasher, s.encoder.Encode(gram))
	}
	_, ok := s.hashes[h]
	return ok
}

// Validate implements barkov.WindowValidator[T], rejecting corpus n-grams.
func (s *HashNGramSet[T]) Validate(gram []T) bool { return !s.Contains(gram) }

// Validator returns a WithValidator callback that rejects corpus n-grams.
func (s *HashNGramSet[T]) Validator() func([]T) bool { return s.Validate }

// Size returns the number of unique hashes in the set.
func (s *HashNGramSet[T]) Size() int { return len(s.hashes) }

// N implements barkov.WindowValidator[T], returning the n-gram width.
func (s *HashNGramSet[T]) N() int { return s.n }

var _ barkov.WindowValidator[string] = (*HashNGramSet[string])(nil)

// hashString hashes the backing bytes of a string without copying.
// Safe because strings are immutable and the hash is computed synchronously.
func hashString(h hashers.Hasher, s string) uint64 {
	if len(s) == 0 {
		return h.Hash(nil)
	}
	return h.Hash(unsafe.Slice(unsafe.StringData(s), len(s)))
}
