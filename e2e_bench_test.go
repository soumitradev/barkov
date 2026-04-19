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
			compressed := barkov.InitChain(4).BuildCompressed(e2eCorpus)
			barkov.Gen(ctx, compressed) //nolint
		}
	})

	b.Run("string/NGramSet", func(b *testing.B) {
		for b.Loop() {
			compressed := barkov.InitChain(4).BuildCompressed(e2eCorpus)
			v := barkov.NewNGramSet(e2eCorpus, n, enc).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("string/FNV", func(b *testing.B) {
		for b.Loop() {
			compressed := barkov.InitChain(4).BuildCompressed(e2eCorpus)
			v := nhash.New(e2eCorpus, n, enc, fnv.FNV{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("string/XXH3", func(b *testing.B) {
		for b.Loop() {
			compressed := barkov.InitChain(4).BuildCompressed(e2eCorpus)
			v := nhash.New(e2eCorpus, n, enc, xxh3.XXH3{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/plain", func(b *testing.B) {
		for b.Loop() {
			vocab := interned.NewVocabulary()
			encoded := vocab.InternCorpus(e2eCorpus)
			compressed := interned.BuildCompressedIndexed(4, encoded)
			barkov.Gen(ctx, compressed) //nolint
		}
	})

	b.Run("interned/NGramSet", func(b *testing.B) {
		for b.Loop() {
			vocab := interned.NewVocabulary()
			encoded := vocab.InternCorpus(e2eCorpus)
			compressed := interned.BuildCompressedIndexed(4, encoded)
			v := barkov.NewNGramSet(e2eCorpusInterned, n, packedEnc).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/FNV", func(b *testing.B) {
		for b.Loop() {
			vocab := interned.NewVocabulary()
			encoded := vocab.InternCorpus(e2eCorpus)
			compressed := interned.BuildCompressedIndexed(4, encoded)
			v := nhash.New(e2eCorpusInterned, n, packedEnc, fnv.FNV{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("interned/XXH3", func(b *testing.B) {
		for b.Loop() {
			vocab := interned.NewVocabulary()
			encoded := vocab.InternCorpus(e2eCorpus)
			compressed := interned.BuildCompressedIndexed(4, encoded)
			v := nhash.New(e2eCorpusInterned, n, packedEnc, xxh3.XXH3{}).Validator()
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})
}

// BenchmarkPipelineSimpleVsMaxOpt compares the plain string pipeline
// against the fully-optimized pipeline at stateSize=4, no validator on
// either side, so the delta isolates the cost of the optimization levers
// (interning, IndexedCompressedChain) rather than mixing in validator work.
// Numbers from this benchmark back the speedup claim in the README.
func BenchmarkPipelineSimpleVsMaxOpt(b *testing.B) {
	ctx := context.Background()

	b.Run("simple", func(b *testing.B) {
		for b.Loop() {
			compressed := barkov.InitChain(4).BuildCompressed(e2eCorpus)
			barkov.Gen(ctx, compressed) //nolint
		}
	})

	b.Run("maxopt", func(b *testing.B) {
		for b.Loop() {
			vocab := interned.NewVocabulary()
			encoded := vocab.InternCorpus(e2eCorpus)
			compressed := interned.BuildCompressedIndexed(4, encoded)
			barkov.Gen(ctx, compressed) //nolint
		}
	})
}

// BenchmarkThreadedVsSequential isolates WithThreaded's effect at
// stateSize=4 with the xxh3 anti-verbatim validator (the case threading
// is meant to help, since rejected candidates force retries). Both
// variants share the same validated Gen config; only the fan-out differs.
func BenchmarkThreadedVsSequential(b *testing.B) {
	ctx := context.Background()
	packedEnc := interned.PackedEncoder{}
	const n = 6

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(e2eCorpus)
	compressed := interned.BuildCompressedIndexed(4, encoded)
	v := nhash.New(encoded, n, packedEnc, xxh3.XXH3{}).Validator()

	b.Run("sequential", func(b *testing.B) {
		for b.Loop() {
			barkov.Gen(ctx, compressed, barkov.WithValidator(v)) //nolint
		}
	})

	b.Run("threaded", func(b *testing.B) {
		for b.Loop() {
			barkov.Gen(ctx, compressed,
				barkov.WithValidator(v),
				barkov.WithThreaded[interned.TokenID](),
			) //nolint
		}
	})
}
