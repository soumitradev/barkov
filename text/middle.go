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
		fwdOpts, fwdHistory, fwdErr := g.middleOptions(g.chain, seed, contextSeed, gc)
		var fwd, bwd, bwdHistory []interned.TokenID
		var bwdErr error
		if forwardOnly {
			if fwdErr == nil {
				fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
			}
		} else {
			bwdOpts, history, err := g.middleOptions(g.reverseChain, revSeed, revSeed, gc)
			bwdHistory = history
			bwdErr = err
			switch {
			case fwdErr == nil && bwdErr == nil:
				fwd, fwdErr, bwd, bwdErr = g.generateHalves(ctx, fwdOpts, bwdOpts)
			case fwdErr == nil:
				fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
			}
		}
		a := g.startAttempt(gc, fwd, fwdErr, bwd, bwdErr)
		if fwdErr != nil {
			g.finishAttempt(gc, a, nil, fwdErr)
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
			g.finishAttempt(gc, a, nil, bwdErr)
			return nil, bwdErr
		}
		walk := middleWalk{seed: len(seed), hidden: len(seed) - len(contextSeed), history: fwdHistory, backwardHistory: bwdHistory}
		if len(fwd) <= len(seed) {
			if gc.seedEnding && !forwardOnly && bwdErr == nil && len(bwd) > len(seed) {
				slices.Reverse(bwd)
				err := g.validateJoined(bwd, gc)
				g.finishAttempt(gc, a, bwd, err)
				if err == nil {
					g.traceWalk(gc, walk, len(bwd)-len(seed))
					return g.vocab.DecodeTokens(bwd), nil
				}
				lastErr = err
				continue // retry the pair; one rejection must not end seed endings
			}
			lastErr = fmt.Errorf("barkov: no forward continuation after the seed: %w", barkov.ErrSentenceFailedValidation)
			g.finishAttempt(gc, a, nil, lastErr)
			forwardOnly = true
			continue
		}
		if !forwardOnly && bwdErr == nil && len(bwd) > len(seed) {
			slices.Reverse(bwd)
			spliced := make([]interned.TokenID, 0, len(bwd)-len(seed)+len(fwd))
			spliced = append(spliced, bwd[:len(bwd)-len(seed)]...)
			spliced = append(spliced, fwd...)
			err := g.validateJoined(spliced, gc)
			if err == nil {
				g.finishAttempt(gc, a, spliced, nil)
				g.traceWalk(gc, walk, len(bwd)-len(seed))
				return g.vocab.DecodeTokens(spliced), nil
			}
			if gc.spliceCheck {
				g.finishAttempt(gc, a, spliced, err)
				lastErr = err
				continue // resample both halves rather than keep only one
			}
		}
		// Validate the complete result, never the temporary or reversed halves.
		// A rejected splice can still leave a useful forward-only response.
		lastErr = g.validateJoined(fwd, gc)
		g.finishAttempt(gc, a, fwd, lastErr)
		if lastErr == nil {
			g.traceWalk(gc, walk, 0)
			return g.vocab.DecodeTokens(fwd), nil
		}
	}
	return nil, lastErr
}

// middleWalk is what an attempt knows of its walk before the result is kept
type middleWalk struct {
	seed, hidden             int
	history, backwardHistory []interned.TokenID // in walk order
}

// traceWalk records the walk of a middle-out result whose seed sits at index
func (g *Generator) traceWalk(gc *genConfig, w middleWalk, index int) {
	if gc.trace == nil {
		return
	}
	backward := g.vocab.DecodeTokens(w.backwardHistory)
	slices.Reverse(backward)
	gc.trace.Walk = Walk{Seed: index, SeedLen: w.seed, Middle: true, Backward: index > 0, HiddenSeed: w.hidden,
		History: g.vocab.DecodeTokens(w.history), BackwardHistory: backward, Unsteered: gc.noSteering || gc.fallback}
}

// middleOptions also returns the history sampled before a short seed, in
// walk order
func (g *Generator) middleOptions(chain interned.BackoffChain, seed, contextSeed []interned.TokenID, gc *genConfig) ([]barkov.GenOption[interned.TokenID], []interned.TokenID, error) {
	history, err := chain.SeedContext(contextSeed)
	if err != nil {
		return nil, nil, err
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
	return []barkov.GenOption[interned.TokenID]{barkov.WithSeed(seed), barkov.WithInitialState(state), barkov.WithMaxTokens[interned.TokenID](gc.maxTokens), barkov.WithParallelism[interned.TokenID](gc.threads)}, history, nil
}

// validateJoined validates an assembled middle-out output, including copied
// runs across the seed when splice validation is on.
func (g *Generator) validateJoined(out []interned.TokenID, gc *genConfig) error {
	if err := g.validateResult(out, gc.maxTokens); err != nil {
		return err
	}
	if gc.noFragments && g.chain.IsFragment(out) {
		return fmt.Errorf("barkov: output is a piece of one corpus message: %w", barkov.ErrSentenceFailedValidation)
	}
	if gc.spliceCheck && g.chain.CopiesRun(out) {
		return fmt.Errorf("barkov: joined output copies a corpus run across the seed: %w", barkov.ErrSentenceFailedValidation)
	}
	return nil
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
