// Provides very basic markov utilities
package main

import (
	"math/rand"
	"os"
	"strings"
)

type Text struct {
	// A container for a markov model
	datamap  map[string][]string // A map of every word, and the words usually follwoing it
	textdata string              // A string that contains the file contents
}

// Generate a markov Text instance given a corpus filepath
func NewLineText(path string) Text {
	txt, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	gen := Text{
		datamap:  make(map[string][]string),
		textdata: string(txt),
	}
	return gen.generateMarkov()

}

// Generate the markov, i.e. populate the datamap of the Text
func (ungenerated *Text) generateMarkov() Text {
	myMap := map[string][]string{}
	lines := strings.Split(ungenerated.textdata, "\n")
	for _, line := range lines {
		words := strings.Split(line, " ")
		for i, word := range words {
			if i < (len(words) - 1) {
				myMap[word] = append(myMap[word], words[i+1])
			}
		}
	}

	ungenerated.datamap = myMap
	return *ungenerated
}

// Generate a sentence with max sentence length (in words), given an optional starting seed (list of words)
func (model *Text) GenerateSentence(maxSentenceLength int, seed ...string) string {
	var generated []string

	switch seedLength := len(seed); {
	case seedLength == 0:
		// Use a random word from the keys if no seed is given
		keys := make([]string, 0, len(model.datamap))
		for k := range model.datamap {
			keys = append(keys, k)
		}
		randIn := rand.Intn(len(keys))
		seed := keys[randIn]

		current := seed
		generated = append(generated, seed)

		// Run through the datamap to make a sentence
		for i := 0; i < maxSentenceLength-1; i++ {
			vals := model.datamap[current]
			valLength := len(vals)
			if valLength == 0 {
				break
			}
			randIn := rand.Intn(len(vals))
			next := vals[randIn]
			generated = append(generated, next)
			current = next
		}
	case seedLength == 1:
		// Use the seed while generatings
		current := seed[0]
		generated = append(generated, current)

		// Run through the datamap to make a sentence
		for i := 0; i < maxSentenceLength-1; i++ {
			vals := model.datamap[current]
			valLength := len(vals)
			if valLength == 0 {
				break
			}
			randIn := rand.Intn(valLength)
			next := vals[randIn]
			generated = append(generated, next)
			current = next
		}
	case seedLength > 1:
		// Use the last part of the seed as the seed, and add the rest to our generated text
		generated = append(generated, seed...)

		current := seed[seedLength-1]

		// Run through the datamap to make a sentence
		for i := 0; i < maxSentenceLength-seedLength; i++ {
			vals := model.datamap[current]
			valLength := len(vals)
			if valLength == 0 {
				break
			}
			randIn := rand.Intn(valLength)
			next := vals[randIn]
			generated = append(generated, next)
			current = next
		}
	}

	return strings.Join(generated, " ")
}
