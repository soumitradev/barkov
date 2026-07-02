package interned

import (
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
)

// The indexedChainN types are thin wrappers over the generic indexedCore.
// Each exists only to give its stateSize a distinct Go type so that
// FastMoverKey[[N]TokenID, TokenID] interfaces stay separate and Gen can
// dispatch on chain.StateSize() to the matching one. All behaviour — Model
// layout, lookup, Move(string), MoveKey, SetRNG, StateSize, Sentinels,
// Encoder — is promoted from the embedded core. They are unexported: Build
// is the only entry point, and callers reach the concrete affordances
// (SetRNG, MoveKey) through the barkov.RNGSettable / barkov.FastMoverKey
// interfaces rather than the concrete types.
type (
	indexedChain1 struct{ *indexedCore[[1]TokenID] }
	indexedChain2 struct{ *indexedCore[[2]TokenID] }
	indexedChain3 struct{ *indexedCore[[3]TokenID] }
	indexedChain4 struct{ *indexedCore[[4]TokenID] }
	indexedChain5 struct{ *indexedCore[[5]TokenID] }
	indexedChain6 struct{ *indexedCore[[6]TokenID] }
	indexedChain7 struct{ *indexedCore[[7]TokenID] }
	indexedChain8 struct{ *indexedCore[[8]TokenID] }
)

// Build constructs the fastest chain representation for stateSize 1..8
// from a pre-interned corpus. The caller is expected to have already run
// the corpus through Vocabulary.InternCorpus.
//
// The returned value implements barkov.GenerativeChain[TokenID]; Gen's
// internal FastMoverKey fast path activates automatically for every
// supported N. Callers who need the concrete affordances assert for them:
//
//	chain := interned.Build(4, encoded)
//	chain.(barkov.RNGSettable).SetRNG(r)                       // deterministic RNG
//	chain.(barkov.FastMoverKey[[4]TokenID, TokenID]).MoveKey(k) // direct MoveKey
//
// Panics on stateSize outside 1..8 — the range matches the FastMoverKey
// dispatch in the core Gen path.
func Build(stateSize int, corpus [][]TokenID) barkov.GenerativeChain[TokenID] {
	switch stateSize {
	case 1:
		return &indexedChain1{buildIndexedCore[[1]TokenID](corpus)}
	case 2:
		return &indexedChain2{buildIndexedCore[[2]TokenID](corpus)}
	case 3:
		return &indexedChain3{buildIndexedCore[[3]TokenID](corpus)}
	case 4:
		return &indexedChain4{buildIndexedCore[[4]TokenID](corpus)}
	case 5:
		return &indexedChain5{buildIndexedCore[[5]TokenID](corpus)}
	case 6:
		return &indexedChain6{buildIndexedCore[[6]TokenID](corpus)}
	case 7:
		return &indexedChain7{buildIndexedCore[[7]TokenID](corpus)}
	case 8:
		return &indexedChain8{buildIndexedCore[[8]TokenID](corpus)}
	default:
		panic(fmt.Sprintf("interned: Build: stateSize %d outside supported range 1..8", stateSize))
	}
}

// Compile-time checks that every variant satisfies GenerativeChain and
// FastMoverKey for its stateSize, and RNGSettable (identical for all N).
var (
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain1)(nil)
	_ barkov.FastMoverKey[[1]TokenID, TokenID] = (*indexedChain1)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain2)(nil)
	_ barkov.FastMoverKey[[2]TokenID, TokenID] = (*indexedChain2)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain3)(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID] = (*indexedChain3)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain4)(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID] = (*indexedChain4)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain5)(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID] = (*indexedChain5)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain6)(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID] = (*indexedChain6)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain7)(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID] = (*indexedChain7)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*indexedChain8)(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID] = (*indexedChain8)(nil)

	_ barkov.RNGSettable = (*indexedChain4)(nil)
)
