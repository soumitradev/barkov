package interned

import barkov "github.com/soumitradev/barkov/v2"

// WithEscape shares immutable tables with the original chain while
// interpolating across context orders
func (b *backoffCore) WithEscape(scale float64) BackoffChain {
	view := *b
	view.escape = max(scale, 0)
	return wrapBackoff(&view)
}

// moveEscape draws exactly from the interpolated mixture over orders under
// copy steering. Order m is picked with weight a_m·Z_m, its escape mixture
// weight times the share of followers steering allows, then a follower from
// its filtered distribution. Renormalising each order alone would skew the
// mixture, e.g. an order left with only END would give END all its weight.
func (b *backoffCore) moveEscape(window []TokenID, sv steerView, probeBuf *[8]TokenID) (TokenID, error) {
	type component struct {
		m      int
		idx    barkov.ChoicesIndex
		weight float64
	}
	var comps [8]component
	n := 0
	failure := noContext
	stay := 1.0 // probability that escaping reaches the current order
	for m := b.cfg.MaxOrder; m >= b.cfg.MinOrder; m-- {
		tail := window[len(window)-m:]
		// A context opening with two Begins repeats the shorter message-start
		// context, so skip it to count start padding once
		if m > b.cfg.MinOrder && tail[0] == b.sentinels.Begin && tail[1] == b.sentinels.Begin {
			continue
		}
		idx, ok := b.tables[m-1].lookupTail(tail)
		if !ok {
			continue
		}
		failure = max(failure, thinContext)
		if idx.Obs < uint32(b.cfg.MinSupport) {
			continue
		}
		failure = allSteered
		e := 0.0
		if m > b.cfg.MinOrder {
			e = b.escapeProbability(m, idx)
		}
		comps[n] = component{m, idx, stay * (1 - e)}
		n++
		stay *= e
	}
	if n > 0 {
		comps[n-1].weight += stay // no shorter order left to escape to
	}
	var total float64
	for i := range n {
		comps[i].weight *= b.allowedShare(comps[i].m, comps[i].idx, window, sv, probeBuf)
		total += comps[i].weight
	}
	if total > 0 {
		pick := b.float64() * total
		for i := range n {
			if pick -= comps[i].weight; pick < 0 || i == n-1 {
				if tok, ok := b.sample(comps[i].m, comps[i].idx, window, sv, probeBuf); ok {
					return tok, nil
				}
			}
		}
	}
	return 0, b.noContinuationErr(failure)
}

// escapeProbability is PPM method C scaled by escape, escape·T/(N+T) capped
// at 1, where T is the state's distinct followers and N its observations
func (b *backoffCore) escapeProbability(m int, idx barkov.ChoicesIndex) float64 {
	t := float64(idx.Count)
	n := float64(idx.Obs)
	if idx.Count > 1 {
		_, cum := b.tables[m-1].slices(idx)
		n = float64(cum[len(cum)-1])
	}
	return min(1, b.escape*t/(n+t))
}

// allowedShare is the fraction of the order-m state's observations that
// copy steering allows after window
func (b *backoffCore) allowedShare(m int, idx barkov.ChoicesIndex, window []TokenID, sv steerView, probeBuf *[8]TokenID) float64 {
	if sv.mode == steerOff {
		return 1
	}
	if idx.Count == 1 {
		if b.filtered(sv, window, TokenID(idx.Offset), probeBuf) {
			return 0
		}
		return 1
	}
	choices, cum := b.tables[m-1].slices(idx)
	var allowed, prev uint32
	for i, tok := range choices {
		if !b.filtered(sv, window, tok, probeBuf) {
			allowed += cum[i] - prev
		}
		prev = cum[i]
	}
	return float64(allowed) / float64(cum[len(cum)-1])
}
