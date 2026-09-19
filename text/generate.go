package text

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

// Generate returns one generated message as tokens. Seed words that survive
// vocabulary lookup lead the output (or sit mid-sentence under SeedMiddle).
// It retries on anti-verbatim or length rejection up to the configured Retries,
// and returns context and barkov errors unchanged.
func (g *Generator) Generate(ctx context.Context, opts ...GenOption) ([]string, error) {
	gc := genConfig{generationConfig: g.defaults}
	for _, opt := range opts {
		opt(&gc)
	}

	if err := gc.generationConfig.validate(g.fixedRNG); err != nil {
		return nil, err
	}

	if gc.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, gc.timeout)
		defer cancel()
	}

	if gc.seedMiddle != nil {
		return g.generateMiddleOut(ctx, &gc)
	}

	seed := g.internSeed(gc.seed)
	return g.generateForward(ctx, &gc, seed)
}

func (g *Generator) generateForward(ctx context.Context, gc *genConfig, seed []interned.TokenID) ([]string, error) {
	genOpts := g.genOptions(seed, gc.maxTokens, gc.threads)

	var lastErr error
	for attempt := 0; attempt < gc.retries; attempt++ {
		out, err := barkov.Gen(ctx, g.chain, genOpts...)
		if err == nil {
			return g.vocab.DecodeTokens(out), nil
		}
		if !errors.Is(err, barkov.ErrSentenceFailedValidation) {
			return nil, err
		}
		lastErr = err
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	return nil, lastErr
}

func (g *Generator) generateMiddleOut(ctx context.Context, gc *genConfig) ([]string, error) {
	if g.reverseChain == nil {
		return nil, errors.New("text: SeedMiddle requires Bidirectional() option at New")
	}
	seed := g.internSeed(gc.seedMiddle)
	if len(seed) == 0 {
		return g.generateForward(ctx, gc, nil)
	}

	revSeed := make([]interned.TokenID, len(seed))
	copy(revSeed, seed)
	slices.Reverse(revSeed)

	fwdOpts := g.genOptions(seed, gc.maxTokens, gc.threads)
	bwdOpts := g.genOptions(revSeed, gc.maxTokens, gc.threads)

	var lastErr error
	for attempt := 0; attempt < gc.retries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		fwd, fwdErr, bwd, bwdErr := g.generateHalves(ctx, fwdOpts, bwdOpts)

		if fwdErr != nil && !errors.Is(fwdErr, barkov.ErrSentenceFailedValidation) {
			return nil, fwdErr
		}
		if bwdErr != nil && !errors.Is(bwdErr, barkov.ErrSentenceFailedValidation) {
			return nil, bwdErr
		}

		var spliced []interned.TokenID
		if fwdErr == nil && bwdErr == nil {
			slices.Reverse(bwd)
			prefixLen := len(bwd) - len(seed)
			if prefixLen < 0 {
				prefixLen = 0
			}
			spliced = make([]interned.TokenID, 0, prefixLen+len(fwd))
			spliced = append(spliced, bwd[:prefixLen]...)
			spliced = append(spliced, fwd...)
		} else if fwdErr == nil {
			// Graceful fallback to forward half if backward failed
			spliced = fwd
		} else if bwdErr == nil {
			// Graceful fallback to backward half if forward failed
			slices.Reverse(bwd)
			spliced = bwd
		} else {
			lastErr = fwdErr
			continue
		}

		if gc.maxTokens > 0 && len(spliced) > gc.maxTokens {
			lastErr = fmt.Errorf("barkov: output exceeded maximum tokens (%d): %w", gc.maxTokens, barkov.ErrSentenceFailedValidation)
			continue
		}
		if g.messages != nil && g.messages.Contains(spliced) {
			lastErr = fmt.Errorf("barkov: output failed validation: %w", barkov.ErrSentenceFailedValidation)
			continue
		}

		return g.vocab.DecodeTokens(spliced), nil
	}
	return nil, lastErr
}

// Sentence is Generate joined with single spaces.
func (g *Generator) Sentence(ctx context.Context, opts ...GenOption) (string, error) {
	out, err := g.Generate(ctx, opts...)
	if err != nil {
		return "", err
	}
	return strings.Join(out, " "), nil
}

// internSeed maps seed words to TokenIDs, dropping any not in the vocabulary
// after falling back to lowercase. Returns nil (unseeded) when nothing survives.
func (g *Generator) internSeed(words []string) []interned.TokenID {
	if len(words) == 0 {
		return nil
	}
	seed := make([]interned.TokenID, 0, len(words))
	for _, w := range words {
		if id, ok := g.vocab.Lookup(w); ok {
			seed = append(seed, id)
			continue
		}
		if lw := strings.ToLower(w); lw != w {
			if id, ok := g.vocab.Lookup(lw); ok {
				seed = append(seed, id)
			}
		}
	}
	if len(seed) == 0 {
		return nil
	}
	return seed
}

// genOptions assembles the core generation options for one attempt.
func (g *Generator) genOptions(seed []interned.TokenID, maxTokens, threads int) []barkov.GenOption[interned.TokenID] {
	opts := make([]barkov.GenOption[interned.TokenID], 0, 4)
	if len(seed) > 0 {
		opts = append(opts, barkov.WithSeed(seed))
	}
	if maxTokens > 0 {
		opts = append(opts, barkov.WithMaxTokens[interned.TokenID](maxTokens))
	}
	if g.messages != nil {
		opts = append(opts, barkov.WithOutputValidator(g.messages.NotWholeMessage()))
	}
	if threads > 0 {
		opts = append(opts, barkov.WithParallelism[interned.TokenID](threads))
	}
	return opts
}

// generateHalves keeps a caller-owned RNG sequential. Independent package RNG
// draws can run concurrently; every started half finishes before returning.
func (g *Generator) generateHalves(ctx context.Context, forward, backward []barkov.GenOption[interned.TokenID]) (fwd []interned.TokenID, fwdErr error, bwd []interned.TokenID, bwdErr error) {
	if g.fixedRNG {
		fwd, fwdErr = barkov.Gen(ctx, g.chain, forward...)
		bwd, bwdErr = barkov.Gen(ctx, g.reverseChain, backward...)
		return
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); bwd, bwdErr = barkov.Gen(ctx, g.reverseChain, backward...) }()
	fwd, fwdErr = barkov.Gen(ctx, g.chain, forward...)
	wg.Wait()
	return
}
