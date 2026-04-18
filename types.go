package barkov

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

// GenerativeChain is the minimum interface needed to generate text.
// All methods are EXPORTED so users in other packages can implement
// their own chain types.
type GenerativeChain[T comparable] interface {
	StateSize() int
	MaxOverlap() int
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

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")
const ErrGenerationTimeout = errorCause("sentence generation timed out")
