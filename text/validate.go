package text

import (
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

// Validate reports whether tokens satisfy the rules Generate imposes on its
// output under opts: at most MaxTokens, not a whole corpus message, no copied
// run as long as the copy-steering window (unless WithoutCopySteering), and
// not a piece of one message under WithoutFragments. Callers that edit a
// generated output use it to keep the same contract. A failure wraps
// barkov.ErrSentenceFailedValidation; words outside the vocabulary match no
// corpus text.
func (g *Generator) Validate(tokens []string, opts ...GenOption) error {
	gc := genConfig{generationConfig: g.defaults}
	for _, opt := range opts {
		opt(&gc)
	}
	ids := make([]interned.TokenID, len(tokens))
	for i, t := range tokens {
		id, ok := g.vocab.Lookup(t)
		if !ok {
			id = g.unknownContext
		}
		ids[i] = id
	}
	if err := g.validateResult(ids, gc.maxTokens); err != nil {
		return err
	}
	if gc.noFragments && g.chain.IsFragment(ids) {
		return fmt.Errorf("barkov: output is a piece of one corpus message: %w", barkov.ErrSentenceFailedValidation)
	}
	if !gc.noSteering && g.chain.CopiesRun(ids) {
		return fmt.Errorf("barkov: output copies a corpus run: %w", barkov.ErrSentenceFailedValidation)
	}
	return nil
}
