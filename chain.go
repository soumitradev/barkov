package barkov

import (
	"runtime"
	"slices"
	"time"
)

func InitChain(contextSize int) *Chain {
	return &Chain{
		stateSize: contextSize,
		model:     make(Model),
	}
}

func calculateCumDist(next map[string]int) ([]string, []int) {
	keys := make([]string, 0, len(next))
	cumDist := make([]int, 0, len(next))

	total := 0
	for k := range next {
		total += next[k]
		keys = append(keys, k)
		cumDist = append(cumDist, total)
	}
	return keys, cumDist
}

func (chain *Chain) precomputeBeginState() {
	beginState := slices.Repeat([]string{BEGIN}, chain.stateSize)
	chain.beginChoices, chain.beginCumDist = calculateCumDist(chain.model[ConstructState(beginState)])
}

func (chain *Chain) Build(corpus [][]string) *Chain {
	beginState := slices.Repeat([]string{BEGIN}, chain.stateSize)

	for _, run := range corpus {
		items := append(beginState, append(run, END)...)
		for i := 0; i < len(run)+1; i++ {
			state := ConstructState(items[i : i+chain.stateSize])
			follow := items[i+chain.stateSize]

			if _, ok := chain.model[state]; !ok {
				chain.model[state] = make(map[string]int)
			}

			if _, ok := chain.model[state][follow]; !ok {
				chain.model[state][follow] = 0
			}

			chain.model[state][follow]++
		}
	}

	chain.precomputeBeginState()
	return chain
}

func (chain *Chain) Compress() *CompressedChain {
	compressedChain := CompressedChain{
		stateSize: chain.stateSize,
		model:     make(CompressedModel),
	}

	for state, choices := range chain.model {
		nextChoices, cumDist := calculateCumDist(choices)
		compressedChain.model[state] = CompressedChoices{
			choices: nextChoices,
			cumDist: cumDist,
		}
	}

	return &compressedChain
}

func (chain *Chain) move(state State) (string, error) {
	if _, ok := chain.model[state]; !ok {
		return "", ErrStateNotFound
	}

	if state == ConstructState(slices.Repeat([]string{BEGIN}, chain.stateSize)) {
		return chain.beginChoices[chooseToken(chain.beginCumDist)], nil
	} else {
		choices, cumDist := calculateCumDist(chain.model[state])
		return choices[chooseToken(cumDist)], nil
	}
}

func (chain *CompressedChain) move(state State) (string, error) {
	if _, ok := chain.model[state]; !ok {
		return "", ErrStateNotFound
	}

	nextChoices := chain.model[state]
	choiceIndex := chooseToken(nextChoices.cumDist)
	return nextChoices.choices[choiceIndex], nil
}

func (chain *Chain) getStateSize() int {
	return chain.stateSize
}

func (chain *CompressedChain) getStateSize() int {
	return chain.stateSize
}

func (chain *Chain) getMaxOverlap() int {
	return chain.stateSize + 2
}

func (chain *CompressedChain) getMaxOverlap() int {
	return chain.stateSize + 2
}

func GenWithStart(chain GenerativeChain, start State) ([]string, error) {
	state := DeconstructState(start)
	generated := initializeGeneration(chain, state)
	state = padState(chain, state)

	for {
		stateString := ConstructState(state)
		next, err := chain.move(stateString)
		if err != nil {
			return nil, err
		}
		if next == END {
			break
		}
		generated = append(generated, next)
		state = append(state[1:], next)
	}
	return generated, nil
}

func Gen(chain GenerativeChain) ([]string, error) {
	return GenWithStart(chain, "")
}

func GenPrunedWithStart(
	chain GenerativeChain,
	start State,
	validGram func([]string) bool,
) ([]string, error) {
	state := DeconstructState(start)
	generated := initializeGeneration(chain, state)
	state = padState(chain, state)

	for {
		stateString := ConstructState(state)
		next, err := chain.move(stateString)
		if err != nil {
			return nil, err
		}
		if next == END {
			break
		}
		generated = append(generated, next)
		if len(generated) >= chain.getMaxOverlap() &&
			!validGram(generated[len(generated)-chain.getMaxOverlap():]) {
			return nil, ErrSentenceFailedValidation
		}
		state = append(state[1:], next)
	}

	if len(generated) < chain.getMaxOverlap() {
		return nil, ErrSentenceTooShort
	}
	return generated, nil
}

func GenPruned(
	chain GenerativeChain,
	validGram func([]string) bool,
) ([]string, error) {
	return GenPrunedWithStart(chain, "", validGram)
}

func GenThreadedWithStart(
	chain GenerativeChain,
	start State,
	validGram func([]string) bool,
	timeout time.Duration,
) ([]string, error) {
	startTime := time.Now()
	found := false
	var final []string
	resChan := make(chan Result)
	MAX_THREADS := runtime.NumCPU() * 8
	for !found {
		for range MAX_THREADS {
			go func() {
				attempt, err := GenPrunedWithStart(chain, start, validGram)
				resChan <- Result{err, attempt}
			}()
		}
		for range MAX_THREADS {
			res := <-resChan
			if res.err != nil {
				if res.err == ErrStateNotFound {
					return nil, res.err
				}
				continue
			}
			found = true
			final = res.words
		}

		currentTime := time.Now()
		if currentTime.Sub(startTime) > timeout {
			return nil, ErrGenerationTimeout
		}
	}
	return final, nil
}

func GenThreaded(
	chain GenerativeChain,
	validGram func([]string) bool,
	timeout time.Duration,
) ([]string, error) {
	return GenThreadedWithStart(chain, "", validGram, timeout)
}
