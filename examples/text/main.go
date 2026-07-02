// The quick start: a corpus of tokenized messages in, non-parroting
// sentences out. The text package picks the fastest engine and turns
// anti-verbatim protection on for you; there is no chain, vocabulary, or
// validator to wire up by hand.
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/soumitradev/barkov/v2/text"
)

func main() {
	corpus := [][]string{
		strings.Fields("the quick brown fox jumps over the lazy dog"),
		strings.Fields("the lazy dog runs after the quick brown fox"),
		strings.Fields("a brown dog and a quick fox are good friends"),
		strings.Fields("the quick fox jumps over a sleepy brown dog"),
		strings.Fields("a lazy fox sleeps while the brown dog runs fast"),
		strings.Fields("the brown fox and the lazy dog play in the sun"),
		strings.Fields("a quick dog jumps over the sleepy lazy fox"),
		strings.Fields("the fox runs fast and the dog sleeps in the sun"),
	}

	// Order 2 keeps the context short so a small corpus can recombine into
	// sentences it never saw. Larger corpora do fine at the default order 4.
	gen, err := text.New(corpus, text.Order(2))
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	for i := 0; i < 3; i++ {
		sentence, err := gen.Sentence(ctx)
		if err != nil {
			fmt.Printf("%d) [no non-verbatim sentence found] %v\n", i+1, err)
			continue
		}
		fmt.Printf("%d) %s\n", i+1, sentence)
	}

	// Seed generation to start from specific words.
	if seeded, err := gen.Sentence(ctx, text.Seed([]string{"a", "lazy"})); err == nil {
		fmt.Printf("seeded) %s\n", seeded)
	} else {
		fmt.Printf("seeded) [no non-verbatim sentence found] %v\n", err)
	}
}
