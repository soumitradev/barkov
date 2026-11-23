package interned

import (
	"iter"
	"math/rand/v2"
	"unsafe"
)

// stateMap is a linear-probing hash table for [N]TokenID keys, which beats
// Go's map on the generation hot path. A zero V marks an empty slot, so
// callers must never insert one; ChoicesIndex never is, since every state
// has a follower.
type stateMap[K comparable, V comparable] struct {
	entries []stateEntry[K, V]
	mask    uint32
	count   uint32
	growAt  uint32 // len(entries)*loadFactorNum/loadFactorDen
	seed    uint64
}

type stateEntry[K comparable, V comparable] struct {
	key K
	val V
}

// lanePrime is the multiply constant from Pelle Evensen's rrmxmx mixer
const lanePrime = 0x9FB21C651E98DF25

// hash folds the key's 8-byte lanes serially so equal lanes can't cancel.
// Snapshots store slot positions, so changing it breaks restoring them.
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

// Grow at 3/5 load, which measured faster than 0.75
const (
	loadFactorNum = 3
	loadFactorDen = 5
)

// newStateMap returns a stateMap that holds sizeHint entries without growing
func newStateMap[K comparable, V comparable](sizeHint int) *stateMap[K, V] {
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

// GetOrSet returns key's value and true, or inserts newVal and returns it
// with false
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

// forEach lets f mutate each live value in place. f must not insert, or
// the table could grow underneath the iteration.
func (m *stateMap[K, V]) forEach(f func(key K, val *V)) {
	var zero V
	for i := range m.entries {
		e := &m.entries[i]
		if e.val != zero {
			f(e.key, &e.val)
		}
	}
}

func (m *stateMap[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		var zero V
		for i := range m.entries {
			if m.entries[i].val != zero {
				if !yield(m.entries[i].key, m.entries[i].val) {
					return
				}
			}
		}
	}
}

func (m *stateMap[K, V]) Len() int { return int(m.count) }
