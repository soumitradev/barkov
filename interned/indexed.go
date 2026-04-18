package interned

import barkov "github.com/soumitradev/barkov/v2"

// IndexedCompressedChain4 is the stateSize=4 variant of the indexed
// compressed chain. It stores the state map with fixed-size [4]TokenID
// keys and pointer-free values, so the entire bucket array is invisible
// to the GC — mark cycles skip it instead of walking string headers.
//
// The type is a thin wrapper over indexedCore; all methods (MoveKey,
// Move, StateSize, MaxOverlap, Sentinels, Encoder, SetRNG) are promoted
// from the embedded core. Construct via BuildCompressedIndexed4.
type IndexedCompressedChain4 struct {
	*indexedCore[[4]TokenID]
}

// IndexedCompressedChain is the original name for the stateSize=4 chain,
// kept as an alias so existing callers compile unchanged. New code that
// wants to be explicit about the stateSize should prefer the N-suffixed
// names.
type IndexedCompressedChain = IndexedCompressedChain4

// BuildCompressedIndexed4 builds the stateSize=4 indexed chain from a
// pre-interned corpus. The caller is expected to have already run the
// corpus through Vocabulary.InternCorpus.
func BuildCompressedIndexed4(corpus [][]TokenID) *IndexedCompressedChain4 {
	return &IndexedCompressedChain4{indexedCore: buildIndexedCore[[4]TokenID](corpus)}
}

// BuildCompressedIndexed is the original stateSize=4 entry point, kept
// as a forwarder so existing callers compile unchanged.
func BuildCompressedIndexed(corpus [][]TokenID) *IndexedCompressedChain4 {
	return BuildCompressedIndexed4(corpus)
}

// Compile-time checks that every variant satisfies both GenerativeChain
// and FastMoverKey for its stateSize.
var (
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain2)(nil)
	_ barkov.FastMoverKey[[2]TokenID, TokenID]     = (*IndexedCompressedChain2)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain3)(nil)
	_ barkov.FastMoverKey[[3]TokenID, TokenID]     = (*IndexedCompressedChain3)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain4)(nil)
	_ barkov.FastMoverKey[[4]TokenID, TokenID]     = (*IndexedCompressedChain4)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain5)(nil)
	_ barkov.FastMoverKey[[5]TokenID, TokenID]     = (*IndexedCompressedChain5)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain6)(nil)
	_ barkov.FastMoverKey[[6]TokenID, TokenID]     = (*IndexedCompressedChain6)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain7)(nil)
	_ barkov.FastMoverKey[[7]TokenID, TokenID]     = (*IndexedCompressedChain7)(nil)
	_ barkov.GenerativeChain[TokenID]              = (*IndexedCompressedChain8)(nil)
	_ barkov.FastMoverKey[[8]TokenID, TokenID]     = (*IndexedCompressedChain8)(nil)
)
