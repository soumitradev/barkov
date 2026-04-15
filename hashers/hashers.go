// Package hashers defines the Hasher interface used by barkov/nhash.
// Import sub-packages (hashers/fnv, hashers/xxh3, hashers/xxhash64) for
// concrete implementations — each sub-package has its own dependencies so
// importing only what you need avoids pulling in unused modules.
package hashers

// Hasher computes a 64-bit hash over a byte slice.
type Hasher interface {
	Hash(data []byte) uint64
}
