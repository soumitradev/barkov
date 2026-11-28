package barkov

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

type Transitioner[T comparable] interface {
	StateSize() int
	Sentinels() Sentinels[T]
	Move(state []T) (T, error)
}

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")
const ErrGenerationTimeout = errorCause("sentence generation timed out")
