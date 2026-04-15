// Package nhash provides a hash-keyed anti-verbatim validator.
// It uses about 50% less memory than barkov.NGramSet by storing uint64
// hashes instead of full string keys. The Hasher is pluggable — pass
// hashers/xxh3.XXH3{} for best performance or hashers/fnv.FNV{} for
// zero extra dependencies.
package nhash

import (
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers"
)

// HashNGramSet is a hash-keyed anti-verbatim validator. Has a per-entry
// false-positive rate of ~1 in 2^64 for high-quality hashers.
type HashNGramSet[T comparable] struct {
	hashes  map[uint64]struct{}
	n       int
	hasher  hashers.Hasher
	encoder barkov.StateEncoder[T]
}

// New builds a HashNGramSet from a corpus. n is the n-gram width; encoder
// converts token slices to byte strings; hasher hashes those byte strings.
func New[T comparable](
	corpus [][]T,
	n int,
	encoder barkov.StateEncoder[T],
	hasher hashers.Hasher,
) *HashNGramSet[T] {
	set := make(map[uint64]struct{}, len(corpus)*4)
	for _, tokens := range corpus {
		for i := 0; i <= len(tokens)-n; i++ {
			key := encoder.Encode(tokens[i : i+n])
			set[hashString(hasher, key)] = struct{}{}
		}
	}
	return &HashNGramSet[T]{hashes: set, n: n, hasher: hasher, encoder: encoder}
}

// Contains reports whether gram is in the set.
func (s *HashNGramSet[T]) Contains(gram []T) bool {
	key := s.encoder.Encode(gram)
	_, ok := s.hashes[hashString(s.hasher, key)]
	return ok
}

// Validator returns a function suitable for WithValidator that rejects any
// gram present in the set.
func (s *HashNGramSet[T]) Validator() func([]T) bool {
	return func(gram []T) bool { return !s.Contains(gram) }
}

// Size returns the number of unique n-grams in the set.
func (s *HashNGramSet[T]) Size() int { return len(s.hashes) }

// hashString hashes the backing bytes of a string without copying.
// Safe because strings are immutable and the hash is computed synchronously.
func hashString(h hashers.Hasher, s string) uint64 {
	if len(s) == 0 {
		return h.Hash(nil)
	}
	return h.Hash(unsafe.Slice(unsafe.StringData(s), len(s)))
}
