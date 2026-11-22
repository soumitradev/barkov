package text

import (
	"fmt"
	"slices"

	"github.com/soumitradev/barkov/v2/interned"
)

// Start and Unknown stand in for non-word context: message start and unseen
// history before a mid-sentence seed.
const (
	Start   = "\x00start"
	Unknown = "\x00unknown"
)

// Walk describes how tokens were walked so Explain can rebuild each step's context.
type Walk struct {
	Seed, SeedLen   int      // seed position and length (0 means unseeded)
	Middle          bool     // walk started from unknown history before a mid-sentence seed
	Backward        bool     // tokens before the seed were walked backwards
	HiddenSeed      int      // leading seed tokens excluded from the starting context
	History         []string // sampled history before a short seed, in message order
	BackwardHistory []string // sampled history after a short seed, in message order
	Unsteered       bool     // copy steering was off for this walk
}

// Choice is one follower a context offers. End marks message end (or start when backward).
type Choice struct {
	Token   string
	End     bool
	Count   uint32
	Steered bool
}

// Context is what one context order offered at a step, with Words in message order.
type Context struct {
	Order        int
	Words        []string
	Found        bool
	Observations uint32
	Supported    bool
	Choices      []Choice // sorted by count, most common first
}

// Step explains one chosen token or stop position (len(tokens) forward, -1 backward).
type Step struct {
	Index    int
	Token    string
	End      bool
	Backward bool
	Contexts []Context // longest order to shortest
	// Used is the sampled order, or 0 when no order offered this token under w
	Used int
}

// Explain rebuilds every chosen step of a longest-context walk (not WithEscape).
// Seed tokens get no step.
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
