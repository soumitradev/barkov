package text

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

func (g *Generator) generateMiddleOut(ctx context.Context, gc *genConfig) ([]string, error) {
	if g.reverseChain == nil {
		return nil, errors.New("text: SeedMiddle requires Bidirectional() option at New")
	}
	seed := g.internSeed(gc.seedMiddle)
	if len(seed) == 0 {
		return g.generateForward(ctx, gc, nil)
	}
	revSeed := slices.Clone(seed)
	slices.Reverse(revSeed)
	var lastErr error
	forwardOnly := false
	for attempt := 0; attempt < gc.retries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		contextSeed := seed
		if forwardOnly && g.messages == nil {
			// Last-resort copying may back off a terminal seed to the shortest
			// permitted context, while still emitting the complete supplied seed.
			contextSeed = seed[max(0, len(seed)-g.minOrder):]
		}
		fwdOpts, fwdErr := g.middleOptions(g.chain, seed, contextSeed, gc)
		var fwd, bwd []interned.TokenID
		var bwdErr error
		if forwardOnly {
			if fwdErr == nil {
				fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
			}
		} else {
			bwdOpts, err := g.middleOptions(g.reverseChain, revSeed, revSeed, gc)
			bwdErr = err
			switch {
			case fwdErr == nil && bwdErr == nil:
				fwd, fwdErr, bwd, bwdErr = g.generateHalves(ctx, fwdOpts, bwdOpts)
			case fwdErr == nil:
				fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
			}
		}
		if fwdErr != nil {
			if !retryableAttempt(fwdErr) {
				return nil, fwdErr
			}
			lastErr = fwdErr
			if bwdErr == nil {
				forwardOnly = true
			}
			continue
		}
		if bwdErr != nil && !retryableAttempt(bwdErr) && !errors.Is(bwdErr, barkov.ErrStateNotFound) {
			return nil, bwdErr
		}
		if len(fwd) <= len(seed) {
			lastErr = fmt.Errorf("barkov: no forward continuation after the seed: %w", barkov.ErrSentenceFailedValidation)
			forwardOnly = true
			continue
		}
		if !forwardOnly && bwdErr == nil && len(bwd) > len(seed) {
			slices.Reverse(bwd)
			spliced := make([]interned.TokenID, 0, len(bwd)-len(seed)+len(fwd))
			spliced = append(spliced, bwd[:len(bwd)-len(seed)]...)
			spliced = append(spliced, fwd...)
			if err := g.validateResult(spliced, gc.maxTokens); err == nil {
				return g.vocab.DecodeTokens(spliced), nil
			}
		}
		// Validate the complete result, never the temporary or reversed halves.
		// A rejected splice can still leave a useful forward-only response.
		lastErr = g.validateResult(fwd, gc.maxTokens)
		if lastErr == nil {
			return g.vocab.DecodeTokens(fwd), nil
		}
	}
	return nil, lastErr
}

func (g *Generator) middleOptions(chain interned.BackoffChain, seed, contextSeed []interned.TokenID, gc *genConfig) ([]barkov.GenOption[interned.TokenID], error) {
	history, err := chain.SeedContext(contextSeed)
	if err != nil {
		return nil, err
	}
	// Unknown history is not a sentence boundary. An ID absent from the
	// trained model makes longer, fabricated contexts miss their lookup.
	state := make([]interned.TokenID, chain.StateSize())
	for i := range state {
		state[i] = g.unknownContext
	}
	known := append(history, contextSeed...)
	if len(known) > len(state) {
		known = known[len(known)-len(state):]
	}
	copy(state[len(state)-len(known):], known)
	return []barkov.GenOption[interned.TokenID]{barkov.WithSeed(seed), barkov.WithInitialState(state), barkov.WithMaxTokens[interned.TokenID](gc.maxTokens), barkov.WithParallelism[interned.TokenID](gc.threads)}, nil
}

func (g *Generator) validateResult(out []interned.TokenID, limit int) error {
	if limit > 0 && len(out) > limit {
		return fmt.Errorf("barkov: output exceeded maximum tokens (%d): %w", limit, barkov.ErrSentenceFailedValidation)
	}
	if g.messages != nil && g.messages.Contains(out) {
		return fmt.Errorf("barkov: output repeats a corpus message: %w", barkov.ErrSentenceFailedValidation)
	}
	return nil
}

// generateHalves keeps a caller-owned RNG sequential. Every started half
// finishes before the call returns, including cancellation and failures.
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
