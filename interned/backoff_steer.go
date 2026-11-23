package interned

func (b *backoffCore) WithoutVerbatim() BackoffChain {
	view := *b
	view.cfg.MaxVerbatim = -1
	return wrapBackoff(&view)
}

func (b *backoffCore) CopiesRun(tokens []TokenID) bool {
	r := b.cfg.MaxVerbatim
	if r <= 0 {
		return false
	}
	for i := 0; i+r <= len(tokens); i++ {
		if _, ok := b.tables[r-1].lookup(keyOf(tokens[i : i+r])); ok {
			return true
		}
	}
	return false
}

// runLongEnough reports whether the context's last MaxVerbatim-1 tokens are
// all real, since only then can a candidate complete a verbatim run
func (b *backoffCore) runLongEnough(key contextKey) bool {
	for _, tok := range key[len(key)-(b.cfg.MaxVerbatim-1):] {
		if tok == b.sentinels.Begin {
			return false
		}
	}
	return true
}

type steerMode uint8

const (
	steerOff    steerMode = iota // nothing to filter this step
	steerSingle                  // one follower to veto
	steerSet                     // small follower set, scanned per candidate
	steerBig                     // large follower set, probed per candidate
)

// steerView holds the followers of the context's last MaxVerbatim-1 tokens.
// Those are exactly the tokens that would complete a corpus MaxVerbatim-gram,
// so one lookup per step replaces a table probe per candidate.
type steerView struct {
	mode   steerMode
	single TokenID   // steerSingle
	set    []TokenID // steerSet, aliases chain storage
}

// steerProbeFallback caps the follower sets scanned by membership. Nearly
// every steering state on prose has one follower.
const steerProbeFallback = 16

func (b *backoffCore) steerSnapshot(key contextKey) steerView {
	r := b.cfg.MaxVerbatim
	idx, ok := b.tables[r-2].lookup(key)
	if !ok {
		return steerView{}
	}
	if idx.Count == 1 {
		return steerView{mode: steerSingle, single: TokenID(idx.Offset)}
	}
	set, _ := b.tables[r-2].slices(idx)
	if len(set) > steerProbeFallback {
		return steerView{mode: steerBig}
	}
	return steerView{mode: steerSet, set: set}
}

// filtered reports whether emitting tok after key would complete a
// verbatim MaxVerbatim-gram
func (b *backoffCore) filtered(sv steerView, key contextKey, tok TokenID) bool {
	// End never appears in an order-R key, but it does appear in follower
	// sets, so it must be let through explicitly for sentences to end
	if tok == b.sentinels.End {
		return false
	}
	switch sv.mode {
	case steerSingle:
		return tok == sv.single
	case steerSet:
		for _, f := range sv.set {
			if f == tok {
				return true
			}
		}
		return false
	case steerBig:
		return b.wouldExtendVerbatim(key, tok)
	}
	return false
}

// wouldExtendVerbatim probes the order-R table, whose keys are the R-grams inside messages
func (b *backoffCore) wouldExtendVerbatim(key contextKey, tok TokenID) bool {
	_, exists := b.tables[b.cfg.MaxVerbatim-1].lookup(key.shift(tok))
	return exists
}
