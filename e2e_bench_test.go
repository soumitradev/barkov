package barkov_test

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/hashers/fnv"
	"github.com/soumitradev/barkov/v2/hashers/xxh3"
	"github.com/soumitradev/barkov/v2/interned"
	"github.com/soumitradev/barkov/v2/nhash"
)

var e2eCorpus [][]string
var e2eCorpusInterned [][]interned.TokenID
var e2eVocab *interned.Vocabulary

func init() {
	f, err := os.Open("testdata/corpus_public.txt")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "***") || strings.HasPrefix(line, "  ") {
			continue
		}
		tokens := strings.Fields(line)
		if len(tokens) >= 4 {
			e2eCorpus = append(e2eCorpus, tokens)
		}
	}

	e2eVocab = interned.NewVocabulary()
	e2eCorpusInterned = e2eVocab.InternCorpus(e2eCorpus)
}

// BenchmarkEndToEndAllConfigs covers every combination of token type and
// validator for end-to-end pipeline comparison across versions.
func BenchmarkEndToEndAllConfigs(b *testing.B) {
	ctx := context.Background()
	enc := barkov.SepEncoder{Sep: barkov.SEP}
	packedEnc := interned.PackedEncoder{}
	const n = 6 // stateSize(4) + 2

	b.Run("string/plain", func(b *testing.B) {
		for b.Loop() {
			chain := barkov.InitChain(4).Build(e2eCorpus)
			compressed := chain.Compress()
			barkov.Gen(ctx, compressed) //nolint
		}
	})

	b.Run("string/NGramSet", func(b *testing.B) {
		for b.Loop() {
			chain := barkov.InitChain(4).Build(e2eCorpus)
			compressed := chain.Compress()
			v := barkov.NewNGramSet(e2eCorpus, n, enc).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("string/FNV", func(b *testing.B) {
		for b.Loop() {
			chain := barkov.InitChain(4).Build(e2eCorpus)
			compressed := chain.Compress()
			v := nhash.New(e2eCorpus, n, enc, fnv.FNV{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("string/XXH3", func(b *testing.B) {
		for b.Loop() {
			chain := barkov.InitChain(4).Build(e2eCorpus)
			compressed := chain.Compress()
			v := nhash.New(e2eCorpus, n, enc, xxh3.XXH3{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/plain", func(b *testing.B) {
		for b.Loop() {
			chain, vocab := interned.InitChain(4)
			encoded := vocab.InternCorpus(e2eCorpus)
			chain.BuildRaw(encoded)
			compressed := chain.Compress()
			barkov.Gen(ctx, compressed) //nolint
		}
	})

	b.Run("interned/NGramSet", func(b *testing.B) {
		for b.Loop() {
			chain, vocab := interned.InitChain(4)
			encoded := vocab.InternCorpus(e2eCorpus)
			chain.BuildRaw(encoded)
			compressed := chain.Compress()
			v := barkov.NewNGramSet(e2eCorpusInterned, n, packedEnc).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/FNV", func(b *testing.B) {
		for b.Loop() {
			chain, vocab := interned.InitChain(4)
			encoded := vocab.InternCorpus(e2eCorpus)
			chain.BuildRaw(encoded)
			compressed := chain.Compress()
			v := nhash.New(e2eCorpusInterned, n, packedEnc, fnv.FNV{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/XXH3", func(b *testing.B) {
		for b.Loop() {
			chain, vocab := interned.InitChain(4)
			encoded := vocab.InternCorpus(e2eCorpus)
			chain.BuildRaw(encoded)
			compressed := chain.Compress()
			v := nhash.New(e2eCorpusInterned, n, packedEnc, xxh3.XXH3{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})
}
