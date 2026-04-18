package interned

import (
	"hash/maphash"
)

// stateMap is a purpose-built open-addressed hash table keyed by a
// pointer-free POD K (always [N]TokenID for N ∈ 2..8). It replaces
// Go's swisstable map on the gen hot path where profiling showed
// internal/runtime/maps.ctrlGroup.matchH2 burning ~36% of total gen
// time per probe. Linear probing with bounded load factor turns each
// lookup into a single contiguous slot read in the common case.
//
// Keys are compared with Go's native == (compiler emits memequal for
// fixed-size PODs). Hashing goes through maphash.Comparable, which
// uses the same runtime.memhash the map would have used anyway; the
// win is in the probe loop, not the hash.
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
	seed   maphash.Seed
}

type stateEntry[K comparable, V comparable] struct {
	key K
	val V
}

// Grow at 1/2 load. Power-of-2 rounding lands the fill-time load factor
// at roughly 0.25-0.45, which keeps expected probe distance near 1.1 —
// the main source of the gen-path win over Go's swisstable. Tightening
// to 0.75 halved the excess Model memory but cost ~6% of the gen win,
// and the gen path is the library's hot loop (build-once, gen-many).
const (
	loadFactorNum = 1
	loadFactorDen = 2
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
		seed:    maphash.MakeSeed(),
	}
	m.growAt = uint32(capacity) * loadFactorNum / loadFactorDen
	return m
}

// Get returns (val, true) if key is present, else (zero, false).
func (m *stateMap[K, V]) Get(key K) (V, bool) {
	h := uint32(maphash.Comparable(m.seed, key))
	i := h & m.mask
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
	h := uint32(maphash.Comparable(m.seed, key))
	i := h & m.mask
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
	h := uint32(maphash.Comparable(m.seed, key))
	i := h & m.mask
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

// Len returns the number of live entries.
func (m *stateMap[K, V]) Len() int { return int(m.count) }
