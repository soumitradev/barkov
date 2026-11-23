package interned

import "math/rand/v2"

// random draws from the RNG set by SetRNG, or from the global source when
// none was set
type random struct{ rng *rand.Rand }

// SetRNG overrides the random source used by the walk. Not safe for
// concurrent use.
func (r *random) SetRNG(rng *rand.Rand) { r.rng = rng }

func (r *random) uint32n(n uint32) uint32 {
	if r.rng != nil {
		return r.rng.Uint32N(n)
	}
	return rand.Uint32N(n)
}

func (r *random) uint64n(n uint64) uint64 {
	if r.rng != nil {
		return r.rng.Uint64N(n)
	}
	return rand.Uint64N(n)
}

func (r *random) float64() float64 {
	if r.rng != nil {
		return r.rng.Float64()
	}
	return rand.Float64()
}
