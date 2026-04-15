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

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")
const ErrGenerationTimeout = errorCause("sentence generation timed out")
