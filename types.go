package barkov

import "math/rand/v2"

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

// GenerativeChain is the minimum interface needed to generate text.
// All methods are EXPORTED so users in other packages can implement
// their own chain types.
//
// Gen computes the validator window as StateSize()+2 by default; that
// policy is fixed here and not a knob on the interface. Use
// WithNGramValidator to override the width per call with a
// WindowValidator that knows its own N().
type GenerativeChain[T comparable] interface {
	StateSize() int
	Sentinels() Sentinels[T]
	Encoder() StateEncoder[T]
	Move(state string) (T, error)
}

// FastMoverKey is an optional optimisation interface for chains with a
// fixed pointer-free state key K (typically [N]T for N ∈ 2..8). When Gen
// detects it, it skips encoder.Encode + Move(string) and calls MoveKey
// directly with the array-shaped key, eliminating one string allocation
// per generated token. genIterSingle dispatches on chain.StateSize() and
// asserts to FastMoverKey[[N]T, T] to pick the matching implementation.
type FastMoverKey[K, T comparable] interface {
	MoveKey(key K) (T, error)
}

// RNGSettable is implemented by chain types whose random source can be
// overridden for deterministic tests and benchmarks. CompressedChain and
// the interned package's indexed chains satisfy it; assert for it when you
// hold a chain as a GenerativeChain interface:
//
//	if s, ok := chain.(barkov.RNGSettable); ok {
//		s.SetRNG(rand.New(rand.NewPCG(1, 2)))
//	}
type RNGSettable interface{ SetRNG(r *rand.Rand) }

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")
const ErrGenerationTimeout = errorCause("sentence generation timed out")
