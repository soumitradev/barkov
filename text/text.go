// Package text builds string generators with backoff, anti-verbatim checks,
// and bounded retries. Supply pre-tokenized corpus messages to New.
package text

import (
	"errors"
	"fmt"
	"iter"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

// Generator is a ready-to-use, string-first text generator. Unless RNG was
// set, its Generate/Sentence methods are safe to call concurrently.
type Generator struct {
	chain          interned.BackoffChain
	reverseChain   interned.BackoffChain
	vocab          *interned.Vocabulary
	defaults       generationConfig
	fixedRNG       bool
	minOrder       int
	unknownContext interned.TokenID // absent from the immutable trained vocabulary

	// messages holds fnv digests of every complete corpus message, so a
	// short output that reproduces a whole message is caught even when it
	// is too short for any n-gram window. nil when anti-verbatim is disabled.
	messages *interned.MessageSet
}

// New builds a Generator from a pre-tokenized corpus (each inner slice is
// one message's tokens; text does no tokenization itself). It builds the
// vocabulary, interns the corpus, and constructs the engine internally:
// a variable-order backoff chain that samples where the corpus genuinely
// branches and steers around verbatim reproductions during the walk. It
// returns an error on an empty or unusable corpus or an invalid option.
func New(corpus [][]string, opts ...Option) (*Generator, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.order < 2 || cfg.order > 8 {
		return nil, fmt.Errorf("text: Order(%d) out of range 2..8", cfg.order)
	}
	if err := cfg.generationConfig.validate(cfg.rng != nil); err != nil {
		return nil, err
	}
	if cfg.antiVerbatim < 0 && cfg.antiVerbatim != avDefault {
		return nil, fmt.Errorf("text: AntiVerbatim(%d) must be >= 0", cfg.antiVerbatim)
	}
	if !hasContent(corpus) {
		return nil, errors.New("text: corpus has no usable messages")
	}

	maxOrder := min(cfg.order+2, 8)
	if cfg.minOrder < 1 || cfg.minOrder > maxOrder {
		return nil, fmt.Errorf("text: MinOrder(%d) out of range 1..%d", cfg.minOrder, maxOrder)
	}
	if cfg.minSupport < 1 || cfg.minSupport > 65535 {
		return nil, fmt.Errorf("text: MinSupport(%d) out of range 1..65535", cfg.minSupport)
	}
	maxVerbatim := -1 // AntiVerbatim(0): steering off
	if cfg.antiVerbatim != 0 {
		window := cfg.antiVerbatim
		if window == avDefault {
			window = cfg.order + 2
		}
		maxVerbatim = max(2, min(window, maxOrder))
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	chain, revChain, messages := buildModels(encoded, interned.BackoffConfig{MaxOrder: maxOrder, MinOrder: cfg.minOrder, MinSupport: cfg.minSupport, MaxVerbatim: maxVerbatim}, cfg.antiVerbatim != 0, cfg.bidirectional)

	g := &Generator{
		chain:          chain,
		reverseChain:   revChain,
		vocab:          vocab,
		defaults:       cfg.generationConfig,
		minOrder:       cfg.minOrder,
		unknownContext: interned.TokenID(vocab.Size()),
		messages:       messages,
	}

	g.SetRNG(cfg.rng)
	return g, nil
}

func hasContent(corpus [][]string) bool {
	for _, msg := range corpus {
		if len(msg) > 0 {
			return true
		}
	}
	return false
}

// Chain returns the underlying engine for callers who need the full core
// API (raw barkov.Gen, custom validators, state inspection). The tokens it
// speaks are interned.TokenID; use Vocab to translate.
func (g *Generator) Chain() barkov.GenerativeChain[interned.TokenID] { return g.chain }

// ReverseChain returns the reverse engine when constructed with Bidirectional(),
// or nil otherwise.
func (g *Generator) ReverseChain() barkov.GenerativeChain[interned.TokenID] { return g.reverseChain }

// Vocab returns the vocabulary that maps between strings and the TokenIDs
// the Chain uses.
func (g *Generator) Vocab() *interned.Vocabulary { return g.vocab }

// Contains reports whether word exists in the vocabulary (tested exact, then lowercased).
func (g *Generator) Contains(word string) bool {
	if g == nil || g.vocab == nil {
		return false
	}
	if _, ok := g.vocab.Lookup(word); ok {
		return true
	}
	if lw := strings.ToLower(word); lw != word {
		if _, ok := g.vocab.Lookup(lw); ok {
			return true
		}
	}
	return false
}

// SetRNG sets the random source for deterministic generation on the generator
// and any reverse engine. Not safe for concurrent use.
func (g *Generator) SetRNG(r *rand.Rand) {
	g.fixedRNG = r != nil
	if s, ok := g.chain.(barkov.RNGSettable); ok {
		s.SetRNG(r)
	}
	if g.reverseChain != nil {
		if s, ok := g.reverseChain.(barkov.RNGSettable); ok {
			s.SetRNG(r)
		}
	}
}

// MessageSet returns the whole-message digest set built from the corpus,
// or nil if AntiVerbatim(0) was set.
func (g *Generator) MessageSet() *interned.MessageSet { return g.messages }

// States iterates all states at the given order (1..MaxOrder) as decoded
// string token slices with their observation counts.
func (g *Generator) States(order int) iter.Seq2[[]string, uint32] {
	bc, ok := g.chain.(interface {
		States(int) iter.Seq2[[]interned.TokenID, uint32]
	})
	if !ok {
		return func(yield func([]string, uint32) bool) {}
	}
	return func(yield func([]string, uint32) bool) {
		for tokens, count := range bc.States(order) {
			words := g.vocab.DecodeTokens(tokens)
			if !yield(words, count) {
				return
			}
		}
	}
}

func buildModels(encoded [][]interned.TokenID, cfg interned.BackoffConfig, antiVerbatim, bidirectional bool) (chain, revChain interned.BackoffChain, messages *interned.MessageSet) {
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		chain = interned.BuildBackoff(cfg, encoded).(interned.BackoffChain)
	}()

	if antiVerbatim {
		wg.Add(1)
		go func() {
			defer wg.Done()
			messages = interned.NewMessageSet(encoded)
		}()
	}

	if bidirectional {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reversedEncoded := make([][]interned.TokenID, len(encoded))
			for i, msg := range encoded {
				rev := make([]interned.TokenID, len(msg))
				copy(rev, msg)
				slices.Reverse(rev)
				reversedEncoded[i] = rev
			}
			revChain = interned.BuildBackoff(cfg, reversedEncoded).(interned.BackoffChain)
		}()
	}

	wg.Wait()

	return
}
