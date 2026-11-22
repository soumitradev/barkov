package text

import (
	"context"
	"errors"
	"strings"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

// Generate returns one generated message as tokens, retrying validation and
// support rejections up to Retries. Context and barkov errors return unchanged.
func (g *Generator) Generate(ctx context.Context, opts ...GenOption) ([]string, error) {
	gc := genConfig{generationConfig: g.defaults}
	for _, opt := range opts {
		opt(&gc)
	}

	if err := gc.generationConfig.validate(g.fixedRNG); err != nil {
		return nil, err
	}

	if gc.trace != nil {
		*gc.trace = Trace{}
	}
	if gc.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, gc.timeout)
		defer cancel()
	}

	g = g.withWalk(&gc)
	out, err := g.generate(ctx, &gc, false)
	if err == nil || !gc.verbatimFallback || g.messages == nil || !errors.Is(err, barkov.ErrSentenceFailedValidation) {
		return out, err
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	fallback := *g
	fallback.chain = g.chain.WithoutVerbatim()
	if g.reverseChain != nil {
		fallback.reverseChain = g.reverseChain.WithoutVerbatim()
	}
	fallback.messages = nil
	return fallback.generate(ctx, &gc, true)
}

// withWalk returns a shallow copy with per-call chain views; tables stay shared.
func (g *Generator) withWalk(gc *genConfig) *Generator {
	if gc.escape <= 0 && !gc.noSteering {
		return g
	}
	view := *g
	for _, chain := range []*interned.BackoffChain{&view.chain, &view.reverseChain} {
		if *chain == nil {
			continue
		}
		if gc.noSteering {
			*chain = (*chain).WithoutVerbatim()
		}
		if gc.escape > 0 {
			*chain = (*chain).WithEscape(gc.escape)
		}
	}
	return &view
}

func (g *Generator) generate(ctx context.Context, gc *genConfig, fallback bool) ([]string, error) {
	if gc.seedMiddle != nil {
		return g.generateMiddleOut(ctx, gc, fallback)
	}
	return g.generateForward(ctx, gc, g.internSeed(gc.seed), fallback)
}

func (g *Generator) generateForward(ctx context.Context, gc *genConfig, seed []interned.TokenID, fallback bool) ([]string, error) {
	genOpts := g.genOptions(seed, gc.maxTokens, gc.threads)
	if gc.noFragments {
		whole := g.messages.NotWholeMessage()
		genOpts = append(genOpts, barkov.WithOutputValidator(func(out []interned.TokenID) bool {
			return whole(out) && !g.chain.IsFragment(out)
		}))
	}

	var lastErr error
	for attempt := 0; attempt < gc.retries; attempt++ {
		out, err := barkov.Gen(ctx, g.chain, genOpts...)
		g.finishAttempt(gc, g.startAttempt(gc, fallback, out, err, nil, nil), out, err)
		if err == nil {
			if gc.trace != nil {
				gc.trace.Walk = Walk{SeedLen: len(seed), Unsteered: gc.noSteering || fallback}
			}
			return g.vocab.DecodeTokens(out), nil
		}
		if !retryableAttempt(err) {
			return nil, err
		}
		lastErr = err
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
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

// internSeed resolves seed words (exact, then lowercased), returning nil if none match.
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

func retryableAttempt(err error) bool {
	return errors.Is(err, barkov.ErrSentenceFailedValidation) || errors.Is(err, barkov.ErrInsufficientSupport)
}
