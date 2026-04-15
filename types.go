package barkov

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

type State = string

type Model = map[State]map[string]int

type CompressedChoices struct {
	CumDist []int
	Choices []string
}

type CompressedModel = map[State]CompressedChoices

type CompressedChain struct {
	Model     CompressedModel
	stateSize int
}

type Chain struct {
	Model        Model
	beginChoices []string
	beginCumDist []int
	stateSize    int
}

type GenerativeChain interface {
	getMaxOverlap() int
	getStateSize() int
	move(state State) (string, error)
}

// GenericGenerativeChain is the minimum interface needed to generate text.
// Unlike the legacy GenerativeChain, this uses EXPORTED methods so users
// in other packages can implement it.
//
// This will be renamed to GenerativeChain in Phase C after the legacy
// interface is removed.
type GenericGenerativeChain[T comparable] interface {
	StateSize() int
	MaxOverlap() int
	Sentinels() Sentinels[T]
	Encoder() StateEncoder[T]
	Move(state string) (T, error)
}

type Result struct {
	err   error
	words []string
}

type errorCause string

func (e errorCause) Error() string {
	return string(e)
}

const ErrStateNotFound = errorCause("state does not exist in model")
const ErrSentenceTooShort = errorCause("generated sentence too short")
const ErrSentenceFailedValidation = errorCause("sentence failed validation")
const ErrGenerationTimeout = errorCause("sentence generation timed out")
