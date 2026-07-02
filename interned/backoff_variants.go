package interned

import barkov "github.com/soumitradev/barkov/v2"

// The backoffChainN types are thin wrappers over backoffCore, mirroring
// the indexedChainN pattern: each exists only to carry the MoveKey method
// with the [N]TokenID key matching StateSize() == MaxOrder == N, so Gen's
// FastMoverKey dispatch picks the fast path. Everything else — Move,
// StateSize, Sentinels, Encoder, SetRNG — is promoted from the embedded
// core. BuildBackoff is the only entry point; callers reach SetRNG through
// barkov.RNGSettable.
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

// Compile-time checks that every wrapper satisfies GenerativeChain,
// FastMoverKey for its MaxOrder, and RNGSettable.
var (
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain2)(nil)
	_ barkov.FastMoverKey[[2]TokenID, TokenID] = (*backoffChain2)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain3)(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID] = (*backoffChain3)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain4)(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID] = (*backoffChain4)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain5)(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID] = (*backoffChain5)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain6)(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID] = (*backoffChain6)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain7)(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID] = (*backoffChain7)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*backoffChain8)(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID] = (*backoffChain8)(nil)

	_ barkov.RNGSettable = (*backoffChain6)(nil)
)
