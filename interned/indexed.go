package interned

import (
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
)

// Build constructs a fixed-order chain for stateSize 1..8 from a corpus
// already run through Vocabulary.InternCorpus. Gen takes the MoveKey fast
// path for every size, and tests reach the RNG through barkov.RNGSettable:
//
//	chain := interned.Build(4, encoded)
//	chain.(barkov.RNGSettable).SetRNG(r)
//
// Panics on stateSize outside 1..8.
func Build(stateSize int, corpus [][]TokenID) barkov.GenerativeChain[TokenID] {
	switch stateSize {
	case 1:
		return buildIndexedCore[[1]TokenID](corpus)
	case 2:
		return buildIndexedCore[[2]TokenID](corpus)
	case 3:
		return buildIndexedCore[[3]TokenID](corpus)
	case 4:
		return buildIndexedCore[[4]TokenID](corpus)
	case 5:
		return buildIndexedCore[[5]TokenID](corpus)
	case 6:
		return buildIndexedCore[[6]TokenID](corpus)
	case 7:
		return buildIndexedCore[[7]TokenID](corpus)
	case 8:
		return buildIndexedCore[[8]TokenID](corpus)
	default:
		panic(fmt.Sprintf("interned: Build: stateSize %d outside supported range 1..8", stateSize))
	}
}

var (
	_ barkov.FastMoverKey[[1]TokenID, TokenID] = (*indexedCore[[1]TokenID])(nil)
	_ barkov.FastMoverKey[[2]TokenID, TokenID] = (*indexedCore[[2]TokenID])(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID] = (*indexedCore[[3]TokenID])(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID] = (*indexedCore[[4]TokenID])(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID] = (*indexedCore[[5]TokenID])(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID] = (*indexedCore[[6]TokenID])(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID] = (*indexedCore[[7]TokenID])(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID] = (*indexedCore[[8]TokenID])(nil)
	_ barkov.RNGSettable                       = (*indexedCore[[4]TokenID])(nil)
)
