package interned

// WithoutVerbatim shares immutable tables and context indexes with the
// original chain while disabling only copy steering
func (b *backoffCore) WithoutVerbatim() BackoffChain {
	view := *b
	view.cfg.MaxVerbatim = -1
	return wrapBackoff(&view)
}

// CopiesRun implements BackoffChain
func (b *backoffCore) CopiesRun(tokens []TokenID) bool {
	r := b.cfg.MaxVerbatim
	if r <= 0 {
		return false
	}
	for i := 0; i+r <= len(tokens); i++ {
		if _, ok := b.tables[r-1].lookupTail(tokens[i : i+r]); ok {
			return true
		}
	}
	return false
}

// runLongEnough reports whether the window's last MaxVerbatim-1 tokens are
// all real, the only case where a candidate can complete a verbatim run.
// The per-window validator likewise only sees full windows.
func (b *backoffCore) runLongEnough(window []TokenID) bool {
	for _, tok := range window[len(window)-(b.cfg.MaxVerbatim-1):] {
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

// steerView holds the followers of the window's last MaxVerbatim-1 tokens.
// Those are exactly the tokens that would complete a corpus MaxVerbatim-gram,
// so one lookup per step replaces a table probe per candidate.
type steerView struct {
	mode   steerMode
	single TokenID   // steerSingle
	set    []TokenID // steerSet, aliases chain storage
}

// steerProbeFallback caps the follower sets scanned by membership. About
// 98% of steering states on prose have one follower, so larger sets are rare.
const steerProbeFallback = 16

func (b *backoffCore) steerSnapshot(window []TokenID) steerView {
	r := b.cfg.MaxVerbatim
	idx, ok := b.tables[r-2].lookupTail(window[len(window)-(r-1):])
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

// filtered reports whether emitting tok after window would complete a
// verbatim MaxVerbatim-gram
func (b *backoffCore) filtered(sv steerView, window []TokenID, tok TokenID, probeBuf *[8]TokenID) bool {
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
		return b.wouldExtendVerbatim(window, tok, probeBuf)
	}
	return false
}

// wouldExtendVerbatim probes the order-R table, whose keys are exactly the
// R-grams found inside messages
func (b *backoffCore) wouldExtendVerbatim(window []TokenID, tok TokenID, probe *[8]TokenID) bool {
	r := b.cfg.MaxVerbatim
	copy(probe[:r-1], window[len(window)-(r-1):])
	probe[r-1] = tok
	_, exists := b.tables[r-1].lookupTail(probe[:r])
	return exists
}
