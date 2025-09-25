package barkov

import (
	"math/rand/v2"
	"slices"
	"sort"
	"strings"
)

func ConstructState(state []string) State {
	return strings.Join(state, SEP)
}

func DeconstructState(state State) []string {
	split := strings.Split(state, SEP)
	if len(split) == 1 && split[0] == state {
		return []string{}
	}
	return split
}

func initializeGeneration(chain GenerativeChain, deconstructed []string) []string {
	generated := make([]string, 0, chain.getStateSize())
	for _, token := range deconstructed {
		if token != BEGIN && token != END {
			generated = append(generated, token)
		}
	}
	return generated
}

func padState(chain GenerativeChain, deconstructed []string) []string {
	startState := make([]string, 0, chain.getStateSize())
	startState = append(startState, slices.Repeat([]string{BEGIN}, max(chain.getStateSize()-len(deconstructed), 0))...)
	startState = append(startState, deconstructed[:min(len(deconstructed), chain.getStateSize())]...)
	return startState
}

func chooseToken(cumDist []int) int {
	choiceNum := rand.IntN(cumDist[len(cumDist)-1])
	return sort.Search(
		len(cumDist), func(i int) bool {
			return cumDist[i] > choiceNum
		},
	)
}
