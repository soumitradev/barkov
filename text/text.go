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
	unknownContext interned.TokenID // absent from the immutable trained vocabulary

	// Whole-message digests catch short verbatim outputs; nil when anti-verbatim is off
	messages *interned.MessageSet
}

// New builds a Generator from pre-tokenized corpus messages using a
// variable-order backoff chain with copy steering.
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
	if cfg.antiVerbatimSet && cfg.antiVerbatim < 0 {
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
	maxVerbatim := -1
	if !cfg.antiVerbatimSet || cfg.antiVerbatim > 0 {
		window := cfg.order + 2
		if cfg.antiVerbatimSet {
			window = cfg.antiVerbatim
		}
		maxVerbatim = max(2, min(window, maxOrder))
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	chain, revChain, messages := buildModels(encoded, interned.BackoffConfig{MaxOrder: maxOrder, MinOrder: cfg.minOrder, MinSupport: cfg.minSupport, MaxVerbatim: maxVerbatim}, maxVerbatim > 0, cfg.bidirectional)

	g := &Generator{
		chain:          chain,
		reverseChain:   revChain,
		vocab:          vocab,
		defaults:       cfg.generationConfig,
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

// Chain returns the forward engine operating on interned.TokenIDs.
func (g *Generator) Chain() barkov.GenerativeChain[interned.TokenID] { return g.chain }

// ReverseChain returns the reverse engine when Bidirectional() was set, else nil.
func (g *Generator) ReverseChain() barkov.GenerativeChain[interned.TokenID] { return g.reverseChain }

// Vocab returns the string-to-TokenID vocabulary shared by the chains.
func (g *Generator) Vocab() *interned.Vocabulary { return g.vocab }

// Contains reports whether word exists in the vocabulary (exact, then lowercased).
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

func (g *Generator) minOrder() int {
	return (interned.ModelSnapshot{Forward: g.chain}).Config().MinOrder
}

// SetRNG sets a deterministic random source on both chains; not concurrency-safe.
func (g *Generator) SetRNG(r *rand.Rand) {
	g.fixedRNG = r != nil
	g.chain.SetRNG(r)
	if g.reverseChain != nil {
		g.reverseChain.SetRNG(r)
	}
}

// MessageSet returns the whole-message digest set, or nil when anti-verbatim is off.
func (g *Generator) MessageSet() *interned.MessageSet { return g.messages }

// States iterates decoded states and observation counts at order 1..MaxOrder.
func (g *Generator) States(order int) iter.Seq2[[]string, uint32] {
	return func(yield func([]string, uint32) bool) {
		for tokens, count := range g.chain.States(order) {
			if !yield(g.vocab.DecodeTokens(tokens), count) {
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
		chain = interned.BuildBackoff(cfg, encoded)
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
			revChain = interned.BuildBackoff(cfg, reversedEncoded)
		}()
	}

	wg.Wait()

	return
}
