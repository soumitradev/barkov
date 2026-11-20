package text

import (
	"slices"

	"github.com/soumitradev/barkov/v2/interned"
)

// Trace records how Generate reached its result
type Trace struct {
	Attempts []Attempt
	// Walk explains the returned tokens; it means nothing when Generate fails
	Walk Walk
}

// Attempt is one try at a message. Forward and Backward are each half in
// message order, nil when the half failed or wasn't walked. Output is the
// result that was checked and Err why the attempt was rejected, nil for the
// attempt Generate returned.
type Attempt struct {
	Forward, Backward       []string
	ForwardErr, BackwardErr error
	Output                  []string
	Err                     error
	// Fallback attempts walked without copy steering after every steered
	// attempt failed
	Fallback bool
}

// WithTrace records every attempt Generate makes into t, replacing what t
// held. Walks inside one attempt still run concurrently.
func WithTrace(t *Trace) GenOption {
	return func(gc *genConfig) { gc.trace = t }
}

// startAttempt returns the attempt for these halves, with bwd in walk order,
// or nil when nothing is traced
func (g *Generator) startAttempt(gc *genConfig, fwd []interned.TokenID, fwdErr error, bwd []interned.TokenID, bwdErr error) *Attempt {
	if gc.trace == nil {
		return nil
	}
	a := &Attempt{ForwardErr: fwdErr, BackwardErr: bwdErr, Fallback: gc.fallback}
	if fwd != nil {
		a.Forward = g.vocab.DecodeTokens(fwd)
	}
	if bwd != nil {
		a.Backward = g.vocab.DecodeTokens(bwd)
		slices.Reverse(a.Backward)
	}
	return a
}

func (g *Generator) finishAttempt(gc *genConfig, a *Attempt, out []interned.TokenID, err error) {
	if a == nil {
		return
	}
	if out != nil {
		a.Output = g.vocab.DecodeTokens(out)
	}
	a.Err = err
	gc.trace.Attempts = append(gc.trace.Attempts, *a)
}
