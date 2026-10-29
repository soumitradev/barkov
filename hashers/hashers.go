// Package hashers defines the Hasher interface used by barkov/nhash.
// Import sub-packages (hashers/fnv, hashers/xxh3, hashers/xxhash64) for
// concrete implementations. Each sub-package has its own dependencies, so
// importing only the one you use keeps the others out of your build.
package hashers

// Hasher computes a 64-bit hash over a byte slice.
type Hasher interface {
	Hash(data []byte) uint64
}
