// Package fnv provides a FNV-1a 64-bit Hasher implementation.
// Zero external dependencies. Suitable when avoiding extra deps matters
// more than raw speed; for best performance use hashers/xxh3 instead.
package fnv

// FNV implements FNV-1a 64-bit hashing.
type FNV struct{}

func (FNV) Hash(data []byte) uint64 {
	h := uint64(0xcbf29ce484222325)
	for _, b := range data {
		h ^= uint64(b)
		h *= 0x100000001b3
	}
	return h
}
