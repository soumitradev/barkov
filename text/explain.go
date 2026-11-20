package text

import (
	"fmt"
	"slices"

	"github.com/soumitradev/barkov/v2/interned"
)

// Start and Unknown stand in for context that isn't a word: the start of a
// message, and the unseen history before a mid-sentence seed
const (
	Start   = "\x00start"
	Unknown = "\x00unknown"
)

// Walk places the seed in generated tokens and says how they were walked, so
// Explain can rebuild the context of every step
type Walk struct {
	// Seed is the seed's index and SeedLen its length; 0 means unseeded
	Seed, SeedLen int
	// Middle walks start from unknown history instead of a message start
	Middle bool
	// Backward means the tokens before the seed were walked backwards from it
	Backward bool
	// HiddenSeed leading seed tokens were left out of the starting context
	HiddenSeed int
	// History is the context sampled before a seed shorter than MinOrder, and
	// BackwardHistory the context after it, both in message order
	History, BackwardHistory []string
	// Unsteered walks had copy steering off
	Unsteered bool
}

// Choice is one follower a context offers. End is the message ending there,
// or for a backward step, starting there.
type Choice struct {
	Token   string
	End     bool
	Count   uint32
	Steered bool
}

// Context is what one context length offered at a step. Words read in
// message order for both directions.
type Context struct {
	Order        int
	Words        []string
	Found        bool
	Observations uint32
	Supported    bool
	// Choices is sorted by count, most common first
	Choices []Choice
}

// Step explains one token a walk chose, or where it stopped
type Step struct {
	// Index is the token's position. A forward walk stops at len(tokens) and
	// a backward walk at -1.
	Index    int
	Token    string
	End      bool
	Backward bool
	// Contexts run from the longest to the shortest
	Contexts []Context
	// Used is the order the walk sampled from, or 0 when none would have
	// offered this token, which means the tokens weren't walked as w says
	Used int
}

// Explain rebuilds each step of a walk that produced tokens: every context
// order the walk tried, what each offered, and which one it sampled from.
// Seed tokens are given, not chosen, so they get no step. It describes the
// longest-context walk, not one under WithEscape.
func (g *Generator) Explain(tokens []string, w Walk) ([]Step, error) {
	if w.Seed < 0 || w.SeedLen < 0 || w.Seed+w.SeedLen > len(tokens) || w.HiddenSeed > w.SeedLen {
		return nil, fmt.Errorf("text: seed %d+%d does not fit %d tokens", w.Seed, w.SeedLen, len(tokens))
	}
	if w.Seed > 0 && !w.Backward {
		return nil, fmt.Errorf("text: tokens before the seed at %d need a backward walk", w.Seed)
	}
	if w.Backward && (w.SeedLen == 0 || g.reverseChain == nil) {
		return nil, fmt.Errorf("text: a backward walk needs a seed and a Bidirectional generator")
	}
	ids := g.lookup(tokens)
	forward, reverse := g.chain, g.reverseChain
	if w.Unsteered {
		forward = forward.WithoutVerbatim()
		if reverse != nil {
			reverse = reverse.WithoutVerbatim()
		}
	}
	pad := forward.Sentinels().Begin
	if w.Middle {
		pad = g.unknownContext
	}
	seedEnd := w.Seed + w.SeedLen
	var steps []Step
	if w.Backward {
		walked := append(slices.Clone(ids[w.Seed:seedEnd]), g.lookup(w.BackwardHistory)...)
		slices.Reverse(walked)
		for i := w.Seed - 1; i >= -1; i-- {
			steps = append(steps, g.explainStep(reverse, pad, walked, tokens, ids, i, true))
			if i >= 0 {
				walked = append(walked, ids[i])
			}
		}
		slices.Reverse(steps)
	}
	walked := append(g.lookup(w.History), ids[w.Seed+w.HiddenSeed:seedEnd]...)
	for i := seedEnd; i <= len(ids); i++ {
		steps = append(steps, g.explainStep(forward, pad, walked, tokens, ids, i, false))
		if i < len(ids) {
			walked = append(walked, ids[i])
		}
	}
	return steps, nil
}

// explainStep explains choosing tokens[i] (or ending, when i is outside
// them) after walked, the tokens so far in walk order
func (g *Generator) explainStep(chain interned.BackoffChain, pad interned.TokenID, walked []interned.TokenID, tokens []string, ids []interned.TokenID, i int, backward bool) Step {
	window := make([]interned.TokenID, chain.StateSize())
	for k := range window {
		window[k] = pad
	}
	copy(window[max(0, len(window)-len(walked)):], walked[max(0, len(walked)-len(window)):])
	end := chain.Sentinels().End
	chosen := end
	step := Step{Index: i, Backward: backward, End: true}
	if i >= 0 && i < len(ids) {
		chosen, step.Token, step.End = ids[i], tokens[i], false
	}
	options, used := chain.Options(window)
	for _, o := range options {
		order := Context{Order: o.Order, Words: g.contextWords(chain, window[len(window)-o.Order:], backward), Found: o.Found, Observations: o.Observations, Supported: o.Supported}
		offered := false
		for k, f := range o.Followers {
			c := Choice{Count: o.Counts[k], Steered: o.Steered[k], End: f == end}
			if !c.End {
				c.Token = g.vocab.Token(f)
			}
			order.Choices = append(order.Choices, c)
			offered = offered || (f == chosen && !c.Steered)
		}
		slices.SortStableFunc(order.Choices, func(a, b Choice) int { return int(b.Count) - int(a.Count) })
		if o.Order == used && offered {
			step.Used = used
		}
		step.Contexts = append(step.Contexts, order)
	}
	return step
}

func (g *Generator) contextWords(chain interned.BackoffChain, window []interned.TokenID, backward bool) []string {
	words := make([]string, len(window))
	for i, id := range window {
		switch id {
		case chain.Sentinels().Begin:
			words[i] = Start
		case g.unknownContext:
			words[i] = Unknown
		default:
			words[i] = g.vocab.Token(id)
		}
	}
	if backward {
		slices.Reverse(words)
	}
	return words
}

// lookup interns words the model knows; the rest match no context
func (g *Generator) lookup(words []string) []interned.TokenID {
	ids := make([]interned.TokenID, len(words))
	for i, w := range words {
		id, ok := g.vocab.Lookup(w)
		if !ok {
			id = g.unknownContext
		}
		ids[i] = id
	}
	return ids
}
