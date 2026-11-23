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

type middleMode uint8

const (
	middleBoth middleMode = iota
	middleForwardOnly
)

type middleSample struct {
	fwd, bwd       []interned.TokenID
	fwdErr, bwdErr error
	walk           middleWalk
}

type middleOutcome struct {
	out      []interned.TokenID
	seedIdx  int
	err      error
	fatal    bool
	nextMode middleMode
}

func (g *Generator) generateMiddleOut(ctx context.Context, gc *genConfig, fallback bool) ([]string, error) {
	if g.reverseChain == nil {
		return nil, errors.New("text: SeedMiddle requires Bidirectional() option at New")
	}
	seed := g.internSeed(gc.seedMiddle)
	if len(seed) == 0 {
		return g.generateForward(ctx, gc, nil, fallback)
	}
	revSeed := slices.Clone(seed)
	slices.Reverse(revSeed)
	var lastErr error
	mode := middleBoth
	for range gc.retries {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		sample := g.sampleHalves(ctx, gc, seed, revSeed, mode)
		a := g.startAttempt(gc, fallback, sample.fwd, sample.fwdErr, sample.bwd, sample.bwdErr)
		res := g.pickCandidate(sample, len(seed), gc, mode)
		g.finishAttempt(gc, a, res.out, res.err)
		if res.err == nil {
			g.traceWalk(gc, fallback, sample.walk, res.seedIdx)
			return g.vocab.DecodeTokens(res.out), nil
		}
		if res.fatal {
			return nil, res.err
		}
		lastErr = res.err
		mode = res.nextMode
	}
	return nil, lastErr
}

func (g *Generator) sampleHalves(ctx context.Context, gc *genConfig, seed, revSeed []interned.TokenID, mode middleMode) middleSample {
	contextSeed := seed
	if mode == middleForwardOnly && g.messages == nil {
		// Last-resort copying may back off a terminal seed to the shortest
		// permitted context, while still emitting the complete supplied seed.
		contextSeed = seed[max(0, len(seed)-g.minOrder()):]
	}
	fwdOpts, fwdHistory, fwdErr := g.middleOptions(g.chain, seed, contextSeed, gc)
	var fwd, bwd, bwdHistory []interned.TokenID
	var bwdErr error
	if mode == middleForwardOnly {
		if fwdErr == nil {
			fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
		}
	} else {
		var bwdOpts []barkov.GenOption[interned.TokenID]
		bwdOpts, bwdHistory, bwdErr = g.middleOptions(g.reverseChain, revSeed, revSeed, gc)
		switch {
		case fwdErr == nil && bwdErr == nil:
			fwd, fwdErr, bwd, bwdErr = g.generateHalves(ctx, fwdOpts, bwdOpts)
		case fwdErr == nil:
			fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
		}
	}
	return middleSample{
		fwd:    fwd,
		bwd:    bwd,
		fwdErr: fwdErr,
		bwdErr: bwdErr,
		walk:   middleWalk{seed: len(seed), hidden: len(seed) - len(contextSeed), history: fwdHistory, backwardHistory: bwdHistory},
	}
}

func (g *Generator) pickCandidate(s middleSample, seedLen int, gc *genConfig, mode middleMode) middleOutcome {
	if s.fwdErr != nil {
		nextMode := mode
		if s.bwdErr == nil {
			nextMode = middleForwardOnly
		}
		return middleOutcome{err: s.fwdErr, fatal: !retryableAttempt(s.fwdErr), nextMode: nextMode}
	}
	if s.bwdErr != nil && !retryableAttempt(s.bwdErr) && !errors.Is(s.bwdErr, barkov.ErrStateNotFound) {
		return middleOutcome{err: s.bwdErr, fatal: true, nextMode: mode}
	}
	hasBackward := mode == middleBoth && s.bwdErr == nil && len(s.bwd) > seedLen
	if len(s.fwd) <= seedLen {
		if gc.seedEnding && hasBackward {
			slices.Reverse(s.bwd)
			err := g.validateJoined(s.bwd, gc)
			return middleOutcome{out: s.bwd, seedIdx: len(s.bwd) - seedLen, err: err, nextMode: mode} // keep middleBoth so one rejection doesn't end seed endings
		}
		err := fmt.Errorf("barkov: no forward continuation after the seed: %w", barkov.ErrSentenceFailedValidation)
		return middleOutcome{err: err, nextMode: middleForwardOnly}
	}
	if hasBackward {
		slices.Reverse(s.bwd)
		spliced := make([]interned.TokenID, 0, len(s.bwd)-seedLen+len(s.fwd))
		spliced = append(spliced, s.bwd[:len(s.bwd)-seedLen]...)
		spliced = append(spliced, s.fwd...)
		err := g.validateJoined(spliced, gc)
		if err == nil {
			return middleOutcome{out: spliced, seedIdx: len(s.bwd) - seedLen, nextMode: mode}
		}
		if gc.spliceCheck {
			return middleOutcome{out: spliced, err: err, nextMode: mode} // resample both halves rather than keep only one
		}
	}
	// Validate the complete result, never the temporary or reversed halves.
	// A rejected splice can still leave a useful forward-only response.
	return middleOutcome{out: s.fwd, err: g.validateJoined(s.fwd, gc), nextMode: mode}
}

type middleWalk struct {
	seed, hidden             int
	history, backwardHistory []interned.TokenID // in walk order
}

func (g *Generator) traceWalk(gc *genConfig, fallback bool, w middleWalk, index int) {
	if gc.trace == nil {
		return
	}
	backward := g.vocab.DecodeTokens(w.backwardHistory)
	slices.Reverse(backward)
	gc.trace.Walk = Walk{Seed: index, SeedLen: w.seed, Middle: true, Backward: index > 0, HiddenSeed: w.hidden,
		History: g.vocab.DecodeTokens(w.history), BackwardHistory: backward, Unsteered: gc.noSteering || fallback}
}

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

func (g *Generator) validateJoined(out []interned.TokenID, gc *genConfig) error {
	if err := g.validateResult(out, gc.maxTokens); err != nil {
		return err
	}
	if gc.noFragments && g.chain.IsFragment(out) {
		return fmt.Errorf("barkov: %w (a piece of one message): %w", barkov.ErrCopiesCorpus, barkov.ErrSentenceFailedValidation)
	}
	if gc.spliceCheck && g.chain.CopiesRun(out) {
		return fmt.Errorf("barkov: %w (a run across the seed): %w", barkov.ErrCopiesCorpus, barkov.ErrSentenceFailedValidation)
	}
	return nil
}

func (g *Generator) validateResult(out []interned.TokenID, limit int) error {
	if limit > 0 && len(out) > limit {
		return fmt.Errorf("barkov: %w (%d): %w", barkov.ErrTokenLimit, limit, barkov.ErrSentenceFailedValidation)
	}
	if g.messages != nil && g.messages.Contains(out) {
		return fmt.Errorf("barkov: %w (a whole message): %w", barkov.ErrCopiesCorpus, barkov.ErrSentenceFailedValidation)
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
