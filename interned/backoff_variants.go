package interned

import barkov "github.com/soumitradev/barkov/v2"

// Each backoffChainN carries the MoveKey whose [N]TokenID key matches
// StateSize() == MaxOrder == N, so Gen's FastMoverKey dispatch finds it
type (
	backoffChain2 struct{ *backoffCore }
	backoffChain3 struct{ *backoffCore }
	backoffChain4 struct{ *backoffCore }
	backoffChain5 struct{ *backoffCore }
	backoffChain6 struct{ *backoffCore }
	backoffChain7 struct{ *backoffCore }
	backoffChain8 struct{ *backoffCore }
)

func (b *backoffChain2) MoveKey(k [2]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain3) MoveKey(k [3]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain4) MoveKey(k [4]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain5) MoveKey(k [5]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain6) MoveKey(k [6]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain7) MoveKey(k [7]TokenID) (TokenID, error) { return b.moveTail(k[:]) }
func (b *backoffChain8) MoveKey(k [8]TokenID) (TokenID, error) { return b.moveTail(k[:]) }

func wrapBackoff(core *backoffCore) BackoffChain {
	switch core.cfg.MaxOrder {
	case 2:
		return &backoffChain2{core}
	case 3:
		return &backoffChain3{core}
	case 4:
		return &backoffChain4{core}
	case 5:
		return &backoffChain5{core}
	case 6:
		return &backoffChain6{core}
	case 7:
		return &backoffChain7{core}
	case 8:
		return &backoffChain8{core}
	default:
		panic("unreachable: validate checked MaxOrder")
	}
}

var (
	_ barkov.FastMoverKey[[2]TokenID, TokenID] = (*backoffChain2)(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID] = (*backoffChain3)(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID] = (*backoffChain4)(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID] = (*backoffChain5)(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID] = (*backoffChain6)(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID] = (*backoffChain7)(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID] = (*backoffChain8)(nil)
)
