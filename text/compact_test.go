package text_test

import (
	"bytes"
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/soumitradev/barkov/v2/text"
)

func TestCompactKeepsGenerationAndSnapshots(t *testing.T) {
	var corpus [][]string
	words := []string{"the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far", "away", "home"}
	rng := rand.New(rand.NewPCG(3, 4))
	for range 400 {
		var m []string
		for range 3 + rng.IntN(8) {
			m = append(m, words[rng.IntN(len(words))])
		}
		corpus = append(corpus, m)
	}
	g, err := text.New(corpus, text.Order(4), text.MinOrder(2), text.AntiVerbatim(4), text.Bidirectional())
	if err != nil {
		t.Fatal(err)
	}
	generate := func() []string {
		g.SetRNG(rand.New(rand.NewPCG(7, 8)))
		var out []string
		for i := range 50 {
			opts := []text.GenOption{text.WithRetries(8)}
			if i%2 == 1 {
				opts = append(opts, text.SeedMiddle([]string{"cat"}))
			}
			tokens, err := g.Generate(context.Background(), opts...)
			out = append(out, fmt.Sprint(tokens, err))
		}
		return out
	}
	before := generate()
	if err := g.Compact(); err != nil {
		t.Fatal(err)
	}
	if after := generate(); !slices.Equal(before, after) {
		t.Fatal("generation changed after Compact")
	}
	var saved bytes.Buffer
	if err := g.Save(&saved); err != nil {
		t.Fatal(err)
	}
	loaded, err := text.Load(bytes.NewReader(saved.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	g = loaded
	if restored := generate(); !slices.Equal(before, restored) {
		t.Fatal("a compacted generator's snapshot generates differently")
	}
}
