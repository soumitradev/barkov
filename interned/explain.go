package interned

// OrderOptions is what one context order offers the walk at a step
type OrderOptions struct {
	Order int
	// Found is false when the corpus never has this context
	Found bool
	// Observations is how often the context was followed by anything.
	// Contexts seen fewer than MinSupport times are skipped.
	Observations uint32
	Supported    bool
	Followers    []TokenID
	Counts       []uint32
	// Steered marks followers copy steering would veto here
	Steered []bool
}

// Options lists what each order from MaxOrder down to MinOrder offers after
// window (length MaxOrder), and the order a walk would sample from: the
// longest supported one with a follower steering allows, or 0 when none has.
// It describes the longest-context walk, not one under WithEscape.
func (b *backoffCore) Options(window []TokenID) ([]OrderOptions, int) {
	var sv steerView
	if b.cfg.MaxVerbatim > 0 && b.runLongEnough(window) {
		sv = b.steerSnapshot(window)
	}
	var probeBuf [8]TokenID
	used := 0
	orders := make([]OrderOptions, 0, b.cfg.MaxOrder-b.cfg.MinOrder+1)
	for m := b.cfg.MaxOrder; m >= b.cfg.MinOrder; m-- {
		o := OrderOptions{Order: m}
		idx, ok := b.tables[m-1].lookupTail(window[len(window)-m:])
		if ok {
			o.Found = true
			if idx.Count == 1 {
				o.Followers, o.Counts = []TokenID{TokenID(idx.Offset)}, []uint32{idx.Obs}
			} else {
				followers, cum := b.tables[m-1].slices(idx)
				o.Followers = followers
				o.Counts = make([]uint32, len(cum))
				prev := uint32(0)
				for i, c := range cum {
					o.Counts[i], prev = c-prev, c
				}
			}
			o.Observations = idx.Obs
			o.Supported = idx.Obs >= uint32(b.cfg.MinSupport)
			o.Steered = make([]bool, len(o.Followers))
			allowed := false
			for i, tok := range o.Followers {
				o.Steered[i] = sv.mode != steerOff && b.filtered(sv, window, tok, &probeBuf)
				allowed = allowed || !o.Steered[i]
			}
			if used == 0 && o.Supported && allowed {
				used = m
			}
		}
		orders = append(orders, o)
	}
	return orders, used
}
