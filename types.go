package barkov

import "math/rand/v2"

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

// GenerativeChain is what Gen needs from a chain, so other packages can supply
// their own. Move may retain its argument and must return promptly so
// cancellation and worker shutdown can finish.
type GenerativeChain[T comparable] interface {
	StateSize() int
	Sentinels() Sentinels[T]
	Encoder() StateEncoder[T]
	Move(state string) (T, error)
}

// FastMoverKey lets Gen pass the state window as K = [StateSize]T, for state
// sizes 1 to 8, skipping the encoder and a string allocation per token.
type FastMoverKey[K, T comparable] interface {
	MoveKey(key K) (T, error)
}

// RNGSettable chains, such as CompressedChain and the interned package's,
// take a seeded random source for deterministic tests and benchmarks:
//
//	if s, ok := chain.(barkov.RNGSettable); ok {
//		s.SetRNG(rand.New(rand.NewPCG(1, 2)))
//	}
type RNGSettable interface{ SetRNG(r *rand.Rand) }

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

// ErrInsufficientSupport means a context exists but has too few observations
const ErrInsufficientSupport = errorCause("insufficient context observations")

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")

// Rejected attempts wrap one of these together with
// ErrSentenceFailedValidation, so callers can test for either
const (
	// ErrTokenLimit means an attempt ran past WithMaxTokens before it ended
	ErrTokenLimit = errorCause("output exceeded maximum tokens")
	// ErrCopiesCorpus means an attempt reproduced training text it was told to avoid
	ErrCopiesCorpus = errorCause("output copies the corpus")
	// ErrWalkCycle means a walk kept repeating the same few tokens, as copy
	// steering can force when every way out of a loop is filtered
	ErrWalkCycle = errorCause("walk repeats a cycle")
)

const ErrGenerationTimeout = errorCause("sentence generation timed out")
