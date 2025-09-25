package barkov

const BEGIN = "</BEGIN/>"
const SEP = "</SEP/>"
const END = "</END/>"

type State = string

type Model = map[State]map[string]int

type CompressedChoices struct {
	cumDist []int
	choices []string
}

type CompressedModel = map[State]CompressedChoices

type CompressedChain struct {
	model     CompressedModel
	stateSize int
}

type Chain struct {
	model        Model
	beginChoices []string
	beginCumDist []int
	stateSize    int
}

type GenerativeChain interface {
	getMaxOverlap() int
	getStateSize() int
	move(state State) (string, error)
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
