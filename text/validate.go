package text

import (
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
)

// Validate checks whether tokens satisfy Generate's output rules under opts
// (MaxTokens, whole-message, copied runs unless WithoutCopySteering, and
// WithoutFragments). It inspects the whole output, so it is stricter than
// Generate's walk-only steering. Failures wrap barkov.ErrSentenceFailedValidation.
func (g *Generator) Validate(tokens []string, opts ...GenOption) error {
	gc := genConfig{generationConfig: g.defaults}
	for _, opt := range opts {
		opt(&gc)
	}
	ids := g.lookup(tokens)
	if err := g.validateResult(ids, gc.maxTokens); err != nil {
		return err
	}
	if gc.noFragments && g.chain.IsFragment(ids) {
		return fmt.Errorf("barkov: %w (a piece of one message): %w", barkov.ErrCopiesCorpus, barkov.ErrSentenceFailedValidation)
	}
	if !gc.noSteering && g.chain.CopiesRun(ids) {
		return fmt.Errorf("barkov: %w (a run): %w", barkov.ErrCopiesCorpus, barkov.ErrSentenceFailedValidation)
	}
	return nil
}
