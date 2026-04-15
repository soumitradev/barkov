package interned_test

import (
	"context"
	"testing"
	"time"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

func TestInternedEndToEnd(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		{"the", "lazy", "dog", "sleeps", "all", "day", "long"},
		{"a", "quick", "brown", "fox", "is", "quick", "and", "brown"},
		{"the", "fox", "jumps", "high", "over", "the", "fence"},
		{"the", "quick", "cat", "jumps", "over", "the", "dog"},
	}

	chain, vocab := interned.InitChain(2)
	encoded := vocab.InternCorpus(corpus)
	chain.BuildRaw(encoded)
	compressed := chain.Compress()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := barkov.Gen(ctx, compressed)
	if err != nil {
		t.Fatalf("gen failed: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("expected non-empty result")
	}

	decoded := vocab.DecodeTokens(result)
	if len(decoded) == 0 {
		t.Error("expected non-empty decoded result")
	}
	t.Logf("Generated: %v", decoded)
}

func TestInternedWithValidator(t *testing.T) {
	corpus := [][]string{
		{"the", "quick", "brown", "fox"},
		{"the", "lazy", "dog"},
		{"a", "quick", "fox", "runs"},
	}

	chain, vocab := interned.InitChain(2)
	encoded := vocab.InternCorpus(corpus)
	chain.BuildRaw(encoded)
	compressed := chain.Compress()

	// Validator that accepts everything
	validator := func(gram []interned.TokenID) bool { return true }

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := barkov.Gen(ctx, compressed, barkov.WithValidator(validator))
	if err != nil {
		t.Fatalf("gen failed: %v", err)
	}

	decoded := vocab.DecodeTokens(result)
	t.Logf("Generated with validator: %v", decoded)
}

func TestPackedEncoderRoundTrip(t *testing.T) {
	enc := interned.PackedEncoder{}
	tokens := []interned.TokenID{0, 1, 42, 100, 999}

	encoded := enc.Encode(tokens)
	decoded := enc.Decode(encoded)

	if len(decoded) != len(tokens) {
		t.Fatalf("length mismatch: got %d want %d", len(decoded), len(tokens))
	}
	for i, id := range tokens {
		if decoded[i] != id {
			t.Errorf("token %d: got %d want %d", i, decoded[i], id)
		}
	}
}

func TestVocabularyInternDecode(t *testing.T) {
	vocab := interned.NewVocabulary()

	id := vocab.Intern("hello")
	if id < interned.FirstUserID {
		t.Errorf("user token got reserved ID %d", id)
	}

	if vocab.Token(id) != "hello" {
		t.Errorf("expected 'hello', got %q", vocab.Token(id))
	}

	// Same token returns same ID
	if vocab.Intern("hello") != id {
		t.Error("expected stable ID for same token")
	}
}
