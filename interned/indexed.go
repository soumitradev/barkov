package interned

import (
	"fmt"

	barkov "github.com/soumitradev/barkov/v2"
)

// IndexedCompressedChain4 is the stateSize=4 variant of the indexed
// compressed chain. It stores the state map with fixed-size [4]TokenID
// keys and pointer-free values, so the entire bucket array is invisible
// to the GC — mark cycles skip it instead of walking string headers.
//
// The type is a thin wrapper over indexedCore; all methods (MoveKey,
// Move, StateSize, Sentinels, Encoder, SetRNG) are promoted from the
// embedded core. Construct via BuildCompressedIndexed4.
type IndexedCompressedChain4 struct {
	*indexedCore[[4]TokenID]
}

// BuildCompressedIndexed4 builds the stateSize=4 indexed chain from a
// pre-interned corpus. The caller is expected to have already run the
// corpus through Vocabulary.InternCorpus.
func BuildCompressedIndexed4(corpus [][]TokenID) *IndexedCompressedChain4 {
	return &IndexedCompressedChain4{indexedCore: buildIndexedCore[[4]TokenID](corpus)}
}

// BuildCompressedIndexed dispatches to the per-N indexed builder that matches
// stateSize. Returns a GenerativeChain[TokenID]; Gen's internal FastMoverKey
// assertion still activates for all supported N via the concrete underlying
// *IndexedCompressedChainN. Callers who want the concrete type (e.g. for
// SetRNG or direct MoveKey) should call BuildCompressedIndexedN directly.
//
// Panics on unsupported stateSize. Supported range is 2..8 — matches the
// FastMoverKey dispatch in the core Gen path.
func BuildCompressedIndexed(stateSize int, corpus [][]TokenID) barkov.GenerativeChain[TokenID] {
	switch stateSize {
	case 2:
		return BuildCompressedIndexed2(corpus)
	case 3:
		return BuildCompressedIndexed3(corpus)
	case 4:
		return BuildCompressedIndexed4(corpus)
	case 5:
		return BuildCompressedIndexed5(corpus)
	case 6:
		return BuildCompressedIndexed6(corpus)
	case 7:
		return BuildCompressedIndexed7(corpus)
	case 8:
		return BuildCompressedIndexed8(corpus)
	default:
		panic(fmt.Sprintf("interned: BuildCompressedIndexed: stateSize %d outside supported range 2..8", stateSize))
	}
}

// Compile-time checks that every variant satisfies both GenerativeChain
// and FastMoverKey for its stateSize.
var (
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain2)(nil)
	_ barkov.FastMoverKey[[2]TokenID, TokenID] = (*IndexedCompressedChain2)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain3)(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID] = (*IndexedCompressedChain3)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain4)(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID] = (*IndexedCompressedChain4)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain5)(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID] = (*IndexedCompressedChain5)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain6)(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID] = (*IndexedCompressedChain6)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain7)(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID] = (*IndexedCompressedChain7)(nil)
	_ barkov.GenerativeChain[TokenID]          = (*IndexedCompressedChain8)(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID] = (*IndexedCompressedChain8)(nil)
)
