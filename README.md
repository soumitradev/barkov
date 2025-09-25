# Barkov

A simple markov chain generator.

Heavily inspired from https://github.com/jsvine/markovify.

## Why?
The reason I made this library is because the markovify library was quite slow, and it did not give me enough control over the tokenization or the validation parts of the markov chain without me having to override the existing classes, which I found very annoying. For this reason, this implementation is quite barebones, and does not come with tokenization or validation code. You can choose to tokenize your text however you want, and validate a sentence in whichever way you see fit. If you don't want to use the chain struct that I've defined, and want to use your own, fine. There's a `GenerativeChain` interface you need to satisfy that has 2 getters and one function to output the next token, and you can use the most useful parts of this library.

Some advantages of this library over the original (in no particular order):
- Much more memory efficient by default (doesn't store too many state variables, relies more on barebones maps and slices)
- You don't need to override the default tokenizer, as there is no default tokenizer
- You don't need to override the default validator, as there is no default validator
- Implements tree pruning during markov generation, to allow for way more efficient generation
- Uses goroutines to peform many generations at a time to allow for faster generation
- Implements a timeout for generation functions that perform validation, allowing for bounded-time generation
- All the useful functions are not written with some chain class in mind, but an interface, allowing for much more customizability

Features that aren't in this library (yet):
- Combining models
- Exporting and importing models to/from JSON

## Usage
This is an exhaustive example for all features of this library.

```go
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/soumitradev/barkov"
)

const STATE_SIZE = 4
const MAX_SENTENCE_LEN = 100
const TIMEOUT = 10 * time.Second

func makeValidator(fullText string) func([]string) bool {
	original := fullText
	// Validator that checks if whatever was generated
	// so far was already in the original text, and
	// rejects if it is.
	return func(gram []string) bool {
		text := strings.Join(gram, " ")
		return !strings.Contains(original, text)
	}
}

func tokenize(messages []string) ([][]string, func([]string) bool) {
	corpus := make([][]string, 0, len(messages))
	var fullText strings.Builder

	for _, message := range messages {
		tokens := strings.Split(message, " ")
		// Filter out messages that are too long or too short
		if len(tokens) < STATE_SIZE || len(tokens) > MAX_SENTENCE_LEN {
			continue
		}

		// Filter out empty tokens that might exist due to multiple spaces
		filtered := make([]string, 0, len(tokens))
		for _, token := range tokens {
			if token == "" {
				continue
			}
			filtered = append(filtered, token)
		}

		corpus = append(corpus, filtered)
		fullText.WriteString(strings.Join(filtered, " ") + "\n")
	}

	return corpus, makeValidator(fullText.String())
}

func readLines(filepath string) []string {
	bytes, err := os.ReadFile(filepath)
	if err != nil {
		panic(fmt.Sprintf("Error reading file at %s", filepath))
	}

	return strings.Split(string(bytes), "\n")
}

func main() {
	filepath := "./corpus.txt"
	messages := readLines(filepath)
	corpus, validator := tokenize(messages)

	fmt.Println("Finished building corpus and context!")
	fmt.Printf("State Size: %d\n", STATE_SIZE)
	chain := barkov.InitChain(STATE_SIZE).Build(corpus).Compress()
	fmt.Println("Finished building and compiling markov model!")

	fmt.Println("Printing 5 random sentences first:")
	for range 5 {
		// Use the threaded version of the generation function with validator and timeout
		generated, err := barkov.GenThreaded(chain, validator, TIMEOUT)
		if err != nil {
			fmt.Println("[ERROR]", err)
			continue
		}
		fmt.Println(strings.Join(generated, " "))
	}

	fmt.Println("Printing 5 random sentences with start states:")
	for range 5 {
		start := barkov.ConstructState([]string{"i", "did", "not"})
		// You can even provide a start state
		generated, err := barkov.GenThreadedWithStart(chain, start, validator, TIMEOUT)
		if err != nil {
			fmt.Println("[ERROR]", err)
			continue
		}
		fmt.Println(strings.Join(generated, " "))
	}
}
```
