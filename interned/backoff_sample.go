package interned

import (
	"sort"

	barkov "github.com/soumitradev/barkov/v2"
)

// sample draws a follower from the order-m state idx. Under steering,
// vetoed candidates are dropped and the rest keep their weights; ok=false
// means every candidate was vetoed.
func (b *backoffCore) sample(m int, idx barkov.ChoicesIndex, window []TokenID, sv steerView, probeBuf *[8]TokenID) (TokenID, bool) {
	if idx.Count == 1 {
		tok := TokenID(idx.Offset)
		if sv.mode != steerOff && b.filtered(sv, window, tok, probeBuf) {
			return 0, false
		}
		return tok, true
	}
	choices, cum := b.tables[m-1].slices(idx)
	switch sv.mode {
	case steerOff:
		return choices[b.draw(cum)], true
	case steerBig:
		return b.sampleReservoir(choices, cum, window, sv, probeBuf)
	default:
		return b.sampleTwoPass(choices, cum, window, sv, probeBuf)
	}
}

// sampleReservoir keeps a weighted reservoir so each costly table probe runs
// once: every survivor replaces the pick with probability weight/totalSoFar
func (b *backoffCore) sampleReservoir(choices []TokenID, cum []uint32, window []TokenID, sv steerView, probeBuf *[8]TokenID) (TokenID, bool) {
	var pick TokenID
	var total, prev uint32
	for i, tok := range choices {
		w := cum[i] - prev
		prev = cum[i]
		if b.filtered(sv, window, tok, probeBuf) {
			continue
		}
		total += w
		if b.uint32n(total) < w {
			pick = tok
		}
	}
	return pick, total > 0
}

// sampleTwoPass totals the survivors' weights and then walks to one draw,
// which needs one RNG call instead of one per survivor
func (b *backoffCore) sampleTwoPass(choices []TokenID, cum []uint32, window []TokenID, sv steerView, probeBuf *[8]TokenID) (TokenID, bool) {
	var total, prev uint32
	for i, tok := range choices {
		w := cum[i] - prev
		prev = cum[i]
		if !b.filtered(sv, window, tok, probeBuf) {
			total += w
		}
	}
	if total == 0 {
		return 0, false
	}
	choiceNum := b.uint32n(total)
	var acc uint32
	prev = 0
	for i, tok := range choices {
		w := cum[i] - prev
		prev = cum[i]
		if b.filtered(sv, window, tok, probeBuf) {
			continue
		}
		if acc += w; acc > choiceNum {
			return tok, true
		}
	}
	panic("unreachable: acc climbs to total > choiceNum")
}

// draw picks an index into a cumulative distribution by weight
func (r *random) draw(cum []uint32) int {
	return scanCumDist(cum, r.uint32n(cum[len(cum)-1]))
}

// scanCumDist returns the first index whose cumulative weight exceeds
// choiceNum. Most states have one or two followers, so a linear scan wins
// until begin-state fanouts in the thousands favour binary search.
func scanCumDist(cum []uint32, choiceNum uint32) int {
	if len(cum) <= 16 {
		for i, c := range cum {
			if c > choiceNum {
				return i
			}
		}
		return len(cum) - 1 // unreachable: choiceNum < cum[len-1]
	}
	return sort.Search(len(cum), func(i int) bool { return cum[i] > choiceNum })
}
