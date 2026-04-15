// Package xxh3 provides an XXH3 64-bit Hasher implementation.
// Fastest and best distribution of the available hashers.
// Requires github.com/zeebo/xxh3.
package xxh3

import "github.com/zeebo/xxh3"

// XXH3 implements 64-bit XXH3 hashing. Recommended default hasher.
type XXH3 struct{}

func (XXH3) Hash(data []byte) uint64 { return xxh3.Hash(data) }
