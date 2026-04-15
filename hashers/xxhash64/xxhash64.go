// Package xxhash64 provides an XXHash64 Hasher implementation.
// Similar speed to XXH3. Requires github.com/cespare/xxhash/v2.
package xxhash64

import "github.com/cespare/xxhash/v2"

// XXHash64 implements 64-bit xxhash hashing.
type XXHash64 struct{}

func (XXHash64) Hash(data []byte) uint64 { return xxhash.Sum64(data) }
