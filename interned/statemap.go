package interned

import (
	"math/rand/v2"
	"unsafe"
)

// stateMap is a purpose-built open-addressed hash table keyed by a
// pointer-free POD K (always [N]TokenID for N ∈ 1..8). It replaces
// Go's swisstable map on the gen hot path where profiling showed
// internal/runtime/maps.ctrlGroup.matchH2 burning ~36% of total gen
// time per probe. Linear probing with bounded load factor turns each
// lookup into a single contiguous slot read in the common case.
//
// Keys are compared with Go's native == (compiler emits memequal for
// fixed-size PODs). Hashing is an inline multiply-xorshift mixer over
// the key's 8-byte lanes (see hash): sizeof(K) is a compile-time
// constant per instantiation, so the mixer compiles to straight-line
// code — no runtime call, unlike maphash.Comparable. The per-map seed
// keeps placement unpredictable across instances. Note the mixer is
// not adversary-grade; that is the right trade for corpus-built
// tables of integer tuples, where multiply mixing is collision-free
// in practice.
//
// V must be comparable and its zero value must be used as the "slot
// empty" sentinel — callers must never insert a zero-valued V. For
// ChoicesIndex this is free: Count==0 is never valid for a real
// state (every state has ≥1 follow token). Dropping an occupied
// bool shrinks [4]TokenID entries from 28→24 bytes (14% denser
// bucket array), measurably reducing cache-miss cost on the gen-
// path probe loop.
type stateMap[K comparable, V comparable] struct {
	entries []stateEntry[K, V]
	mask    uint32
	count   uint32
	// growAt is precomputed len(entries)*loadFactorNum/loadFactorDen so
	// the hot-path grow check is a single uint32 compare.
	growAt uint32
	seed   uint64
}

type stateEntry[K comparable, V comparable] struct {
	key K
	val V
}

// lanePrime is the folded-multiply constant shared by wyhash/romu-style
// mixers; odd and bit-balanced, it gives full 64-bit avalanche per lane.
const lanePrime = 0x9FB21C651E98DF25

// hash mixes key into a probe hash. The lanes are folded serially
// (h ^= lane; h *= lanePrime) so equal lanes at different positions
// cannot cancel, and a final xor-multiply-xor avalanche spreads entropy
// into the low bits the probe mask consumes. The n >= size guards are
// compile-time constants per K instantiation, so this compiles to
// straight-line code with one multiply per 8 bytes of key.
func (m *stateMap[K, V]) hash(key K) uint64 {
	n := int(unsafe.Sizeof(key))
	p := unsafe.Pointer(&key)
	h := m.seed
	if n >= 8 {
		h = (h ^ *(*uint64)(p)) * lanePrime
	}
	if n >= 16 {
		h = (h ^ *(*uint64)(unsafe.Add(p, 8))) * lanePrime
	}
	if n >= 24 {
		h = (h ^ *(*uint64)(unsafe.Add(p, 16))) * lanePrime
	}
	if n >= 32 {
		h = (h ^ *(*uint64)(unsafe.Add(p, 24))) * lanePrime
	}
	if n%8 != 0 { // trailing 4-byte lane: n ∈ {4, 12, 20, 28}
		h = (h ^ uint64(*(*uint32)(unsafe.Add(p, n-4)))) * lanePrime
	}
	h ^= h >> 29
	h *= lanePrime
	h ^= h >> 32
	return h
}

// Grow at 3/5 load. A prior 0.75 attempt cost ~6% of the gen win, but
// profile showed Get's cost is ~88% memory latency on the first bucket
// load — shrinking the table wins back cache residency that outweighs
// the ~1 extra probe distance at 0.6 vs 0.5. Power-of-2 rounding lands
// the fill-time load factor at ~0.3–0.55.
const (
	loadFactorNum = 3
	loadFactorDen = 5
)

// newStateMap returns a stateMap sized so that `sizeHint` entries
// fit under the load-factor threshold without a resize.
func newStateMap[K comparable, V comparable](sizeHint int) *stateMap[K, V] {
	// Need capacity such that sizeHint < capacity*loadFactorNum/loadFactorDen.
	minCap := (sizeHint*loadFactorDen + loadFactorNum - 1) / loadFactorNum
	capacity := 16
	for capacity <= minCap {
		capacity <<= 1
	}
	m := &stateMap[K, V]{
		entries: make([]stateEntry[K, V], capacity),
		mask:    uint32(capacity - 1),
		seed:    rand.Uint64(),
	}
	m.growAt = uint32(capacity) * loadFactorNum / loadFactorDen
	return m
}

// Get returns (val, true) if key is present, else (zero, false).
func (m *stateMap[K, V]) Get(key K) (V, bool) {
	i := uint32(m.hash(key)) & m.mask
	var zero V
	for {
		e := &m.entries[i]
		if e.val == zero {
			return zero, false
		}
		if e.key == key {
			return e.val, true
		}
		i = (i + 1) & m.mask
	}
}

// GetOrSet returns the existing value for key if present, else inserts
// newVal and returns (newVal, false). Lets callers avoid a double
// probe on the build path's "check-then-insert" pattern.
func (m *stateMap[K, V]) GetOrSet(key K, newVal V) (V, bool) {
	if m.count >= m.growAt {
		m.grow()
	}
	i := uint32(m.hash(key)) & m.mask
	var zero V
	for {
		e := &m.entries[i]
		if e.val == zero {
			e.key = key
			e.val = newVal
			m.count++
			return newVal, false
		}
		if e.key == key {
			return e.val, true
		}
		i = (i + 1) & m.mask
	}
}

// Put overwrites any existing entry for key.
func (m *stateMap[K, V]) Put(key K, val V) {
	if m.count >= m.growAt {
		m.grow()
	}
	m.putNoGrow(key, val)
}

func (m *stateMap[K, V]) putNoGrow(key K, val V) {
	i := uint32(m.hash(key)) & m.mask
	var zero V
	for {
		e := &m.entries[i]
		if e.val == zero {
			e.key = key
			e.val = val
			m.count++
			return
		}
		if e.key == key {
			e.val = val
			return
		}
		i = (i + 1) & m.mask
	}
}

func (m *stateMap[K, V]) grow() {
	old := m.entries
	newCap := len(old) * 2
	m.entries = make([]stateEntry[K, V], newCap)
	m.mask = uint32(newCap - 1)
	m.growAt = uint32(newCap) * loadFactorNum / loadFactorDen
	m.count = 0
	var zero V
	for i := range old {
		if old[i].val != zero {
			m.putNoGrow(old[i].key, old[i].val)
		}
	}
}

// forEach calls f once per live entry, in slot order. f receives a
// pointer to the stored value and may mutate it in place; it must not
// insert or the table may grow underneath the iteration.
func (m *stateMap[K, V]) forEach(f func(key K, val *V)) {
	var zero V
	for i := range m.entries {
		e := &m.entries[i]
		if e.val != zero {
			f(e.key, &e.val)
		}
	}
}

// Len returns the number of live entries.
func (m *stateMap[K, V]) Len() int { return int(m.count) }
