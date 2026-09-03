package interned

import (
	"github.com/soumitradev/barkov/v2/hashers/fnv"
)

// MessageSet stores 64-bit FNV digests of complete token sequences, enabling
// O(1) checks for exact-match reproduction of any corpus message regardless
// of length. Safe for concurrent lookups after construction.
type MessageSet struct {
	digests map[uint64]struct{}
	encoder PackedEncoder
}

// NewMessageSet builds a MessageSet from pre-interned corpus messages.
func NewMessageSet(encoded [][]TokenID) *MessageSet {
	set := make(map[uint64]struct{}, len(encoded))
	enc := PackedEncoder{}
	var buf []byte
	for _, msg := range encoded {
		buf = enc.AppendEncoded(buf[:0], msg)
		set[fnv.FNV{}.Hash(buf)] = struct{}{}
	}
	return &MessageSet{
		digests: set,
		encoder: enc,
	}
}

// Contains reports whether tokens matches a complete corpus message in the set.
func (ms *MessageSet) Contains(tokens []TokenID) bool {
	if ms == nil || len(ms.digests) == 0 {
		return false
	}
	var buf [256]byte
	b := ms.encoder.AppendEncoded(buf[:0], tokens)
	_, found := ms.digests[fnv.FNV{}.Hash(b)]
	return found
}

// NotWholeMessage returns a validator function suitable for barkov.WithOutputValidator.
// It returns false (rejects) if the output is an exact match for a whole message in the set.
func (ms *MessageSet) NotWholeMessage() func([]TokenID) bool {
	return func(out []TokenID) bool {
		return !ms.Contains(out)
	}
}

// Size returns the number of unique message digests in the set.
func (ms *MessageSet) Size() int {
	if ms == nil {
		return 0
	}
	return len(ms.digests)
}
