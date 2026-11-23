package barkov

import "unsafe"

// keyArena hands out map keys carved from shared chunks, so N keys cost
// O(log N) allocations. Retired chunks stay alive through the keys that reference them.
type keyArena struct {
	chunks [][]byte
	cur    []byte
}

func newKeyArena(initialCap int) *keyArena {
	if initialCap < 4096 {
		initialCap = 4096
	}
	return &keyArena{cur: make([]byte, 0, initialCap)}
}

func (a *keyArena) Append(b []byte) string {
	need := len(b)
	if need == 0 {
		return ""
	}
	if cap(a.cur)-len(a.cur) < need {
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
	// Safe because arena bytes are never written again once handed out
	return unsafe.String(&a.cur[start], need)
}

// keyStream encodes windows into map keys for corpus scans. A probe key is
// only valid until the next probe, so intern must directly follow the probe
// whose key it keeps.
type keyStream[T comparable] struct {
	appendEnc AppendEncoder[T]
	encoder   StateEncoder[T]
	hasAppend bool
	scratch   []byte
	arena     *keyArena
}

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

func (ks *keyStream[T]) probe(window []T) string {
	if ks.hasAppend {
		ks.scratch = ks.scratch[:0]
		ks.scratch = ks.appendEnc.AppendEncoded(ks.scratch, window)
		// Aliases scratch, which the next probe overwrites
		return unsafe.String(unsafe.SliceData(ks.scratch), len(ks.scratch))
	}
	return ks.encoder.Encode(window)
}

func (ks *keyStream[T]) intern(probeKey string) string {
	if ks.hasAppend {
		return ks.arena.Append(ks.scratch)
	}
	return probeKey
}
