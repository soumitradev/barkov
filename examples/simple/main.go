// Tier 1: drop-in string chain. No interning, no custom hashers, no
// validator. The fastest path to "give me a markov chain" — three lines
// of setup, then generate.
package main

import (
	"context"
	"fmt"
	"strings"

	barkov "github.com/soumitradev/barkov/v2"
)

func main() {
	corpus := [][]string{
		strings.Fields("the quick brown fox jumps over the lazy dog"),
		strings.Fields("the quick brown fox jumps again and again today"),
		strings.Fields("the lazy dog sleeps all day in the warm sun"),
		strings.Fields("the quick fox runs fast and smart across the field"),
		strings.Fields("a quick brown fox is faster than a lazy dog today"),
		strings.Fields("the fox and the dog are good friends in the forest"),
		strings.Fields("the brown dog barks loudly at the quick fox nearby"),
		strings.Fields("a lazy fox sleeps under the warm sun all day long"),
	}

	chain := barkov.InitChain(2).BuildCompressed(corpus)

	fmt.Println("Tier 1 — string chain, no validator:")
	for i := range 5 {
		out, err := barkov.Gen(context.Background(), chain)
		if err != nil {
			fmt.Printf("%d) [error] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, strings.Join(out, " "))
	}
}
