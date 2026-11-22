package barkov

import "unsafe"

// keyArena is a growable byte slab that hands out non-aliasing substrings
// for use as map keys. A new chunk is allocated only when the current one
// cannot fit an appended key; old chunks remain alive because the strings
// returned by Append reference them. The arena therefore produces N
// immutable strings using O(log N) allocations regardless of N, replacing
// the N per-key allocations of a naïve per-call encoder.
type keyArena struct {
	chunks [][]byte // retired chunks kept alive by outstanding string refs
	cur    []byte   // active chunk; we only ever append within its cap
}

func newKeyArena(initialCap int) *keyArena {
	if initialCap < 4096 {
		initialCap = 4096
	}
	return &keyArena{cur: make([]byte, 0, initialCap)}
}

// Append records b as an immutable string key backed by the arena. The
// returned string's backing storage stays valid for the arena's lifetime
// because we never overwrite arena bytes once published.
func (a *keyArena) Append(b []byte) string {
	need := len(b)
	if need == 0 {
		return ""
	}
	if cap(a.cur)-len(a.cur) < need {
		// Retire the current chunk; start a new one big enough for this
		// key and large enough to amortise future appends.
		a.chunks = append(a.chunks, a.cur)
		newCap := cap(a.cur) * 2
		if newCap < need {
			newCap = need * 2
		}
		if newCap < 4096 {
			newCap = 4096
		}
		a.cur = make([]byte, 0, newCap)
	}
	start := len(a.cur)
	a.cur = append(a.cur, b...)
	return unsafe.String(&a.cur[start], need)
}

// keyStream wraps the arena + append-encoder plumbing that every
// string-keyed corpus scan uses (BuildRaw, BuildCompressed, NGramSet).
// Callers call probe(window) to get an ephemeral map key and, on a
// cache miss, intern(probeKey) to upgrade it to a stable heap-backed
// key. When the encoder does not implement AppendEncoder, probe falls
// back to the per-call Encode path and intern is a no-op.
//
// It holds no per-observation state beyond the scratch buffer, so each
// probe call overwrites the previous probe's bytes. Callers must not
// reuse an old probe string after a subsequent probe.
type keyStream[T comparable] struct {
	appendEnc AppendEncoder[T]
	encoder   StateEncoder[T]
	hasAppend bool
	scratch   []byte
	arena     *keyArena
}

// newKeyStream builds a stream for encoder. arenaCapHint is the initial
// capacity the arena should size itself for; honoured only when the
// encoder supplies an AppendEncoder fast path.
func newKeyStream[T comparable](encoder StateEncoder[T], arenaCapHint int) *keyStream[T] {
	appendEnc, hasAppend := any(encoder).(AppendEncoder[T])
	ks := &keyStream[T]{
		encoder:   encoder,
		appendEnc: appendEnc,
		hasAppend: hasAppend,
	}
	if hasAppend {
		ks.scratch = make([]byte, 0, 256)
		ks.arena = newKeyArena(arenaCapHint)
	}
	return ks
}

// probe returns a string key for the given window. In the AppendEncoder
// fast path the bytes live in an internal scratch buffer and are valid
// only until the next probe call; use the string just long enough to
// probe a map, and call intern to keep a copy. In the plain-encoder
// fallback the returned string is already heap-backed.
func (ks *keyStream[T]) probe(window []T) string {
	if ks.hasAppend {
		ks.scratch = ks.scratch[:0]
		ks.scratch = ks.appendEnc.AppendEncoded(ks.scratch, window)
		return unsafe.String(unsafe.SliceData(ks.scratch), len(ks.scratch))
	}
	return ks.encoder.Encode(window)
}

// intern promotes the current probe bytes to a stable, heap-backed key
// safe to store long-term. Pass the most recent probe return value as
// probeKey; when the plain-encoder path is active intern is a no-op
// and returns probeKey unchanged.
func (ks *keyStream[T]) intern(probeKey string) string {
	if ks.hasAppend {
		return ks.arena.Append(ks.scratch)
	}
	return probeKey
}
