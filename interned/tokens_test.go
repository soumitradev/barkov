package interned_test

import (
	"context"
	"slices"
	"testing"

	"github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

func TestInternReturnsSameIDForSameWord(t *testing.T) {
	vocab := interned.NewVocabulary()
	first := vocab.Intern("fox")
	vocab.Intern("dog")
	if again := vocab.Intern("fox"); again != first {
		t.Fatalf("Intern(fox) gave %d then %d", first, again)
	}
}

func TestWordReturnsInternedWord(t *testing.T) {
	vocab := interned.NewVocabulary()
	for _, word := range []string{"the", "quick", "brown", "fox"} {
		if got, ok := vocab.Word(vocab.Intern(word)); got != word || !ok {
			t.Errorf("Word(Intern(%q)) = %q", word, got)
		}
	}
}

func TestInternNeverHandsOutSentinelIDs(t *testing.T) {
	vocab := interned.NewVocabulary()
	// The literal sentinel strings can show up in real chat, so they must get ordinary IDs
	for _, word := range []string{"first", barkov.BEGIN, barkov.END} {
		if id := vocab.Intern(word); id == interned.BEGIN_TOKEN || id == interned.END_TOKEN {
			t.Errorf("Intern(%q) = %d, a sentinel ID", word, id)
		}
	}
}

func TestIDDoesNotInternUnknownWords(t *testing.T) {
	vocab := interned.NewVocabulary()
	if _, ok := vocab.ID("ghost"); ok {
		t.Fatal("ID found a word that was never interned")
	}
	if _, ok := vocab.ID("ghost"); ok {
		t.Fatal("ID added the word it was asked about")
	}
}

func TestInternCorpusMapsBackToCorpus(t *testing.T) {
	corpus := [][]string{{"the", "quick", "fox"}, {"the", "dog"}}
	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	if len(encoded) != len(corpus) {
		t.Fatalf("got %d sentences, want %d", len(encoded), len(corpus))
	}
	for i, ids := range encoded {
		words := make([]string, len(ids))
		for j, id := range ids {
			word, ok := vocab.Word(id)
			if !ok {
				t.Fatalf("sentence %d has id %d with no word", i, id)
			}
			words[j] = word
		}
		if !slices.Equal(words, corpus[i]) {
			t.Errorf("sentence %d decodes to %v, want %v", i, words, corpus[i])
		}
	}
}

func TestInternedChainGeneratesCorpusWords(t *testing.T) {
	corpus := [][]string{{"the", "quick", "brown", "fox"}, {"the", "lazy", "dog"}}
	vocab := interned.NewVocabulary()
	chain := interned.Build(2, vocab.InternCorpus(corpus))

	ids, err := barkov.Gen(context.Background(), chain)
	if err != nil {
		t.Fatalf("Gen: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("Gen returned no tokens")
	}
	for _, id := range ids {
		word, ok := vocab.Word(id)
		if !ok {
			t.Fatalf("generated id %d has no word", id)
		}
		if !slices.ContainsFunc(corpus, func(s []string) bool { return slices.Contains(s, word) }) {
			t.Errorf("generated %q (id %d), which isn't a corpus word", word, id)
		}
	}
}
