package interned

// The per-N variants below are thin wrappers over the generic indexedCore.
// Each type exists only to give its stateSize a distinct Go type so that
// FastMoverKey[[N]TokenID, TokenID] interfaces stay separate and Gen can
// dispatch on chain.StateSize() to the matching one. All behaviour —
// Model layout, lookup, Move(string), MoveKey, SetRNG, StateSize,
// MaxOverlap, Sentinels, Encoder — is promoted from the embedded core.
//
// IndexedCompressedChain4 and its BuildCompressedIndexed4 live in
// indexed.go (plus the IndexedCompressedChain/BuildCompressedIndexed
// backward-compat aliases); everything else is here.

type IndexedCompressedChain2 struct{ *indexedCore[[2]TokenID] }
type IndexedCompressedChain3 struct{ *indexedCore[[3]TokenID] }
type IndexedCompressedChain5 struct{ *indexedCore[[5]TokenID] }
type IndexedCompressedChain6 struct{ *indexedCore[[6]TokenID] }
type IndexedCompressedChain7 struct{ *indexedCore[[7]TokenID] }
type IndexedCompressedChain8 struct{ *indexedCore[[8]TokenID] }

func BuildCompressedIndexed2(corpus [][]TokenID) *IndexedCompressedChain2 {
	return &IndexedCompressedChain2{indexedCore: buildIndexedCore[[2]TokenID](corpus)}
}

func BuildCompressedIndexed3(corpus [][]TokenID) *IndexedCompressedChain3 {
	return &IndexedCompressedChain3{indexedCore: buildIndexedCore[[3]TokenID](corpus)}
}

func BuildCompressedIndexed5(corpus [][]TokenID) *IndexedCompressedChain5 {
	return &IndexedCompressedChain5{indexedCore: buildIndexedCore[[5]TokenID](corpus)}
}

func BuildCompressedIndexed6(corpus [][]TokenID) *IndexedCompressedChain6 {
	return &IndexedCompressedChain6{indexedCore: buildIndexedCore[[6]TokenID](corpus)}
}

func BuildCompressedIndexed7(corpus [][]TokenID) *IndexedCompressedChain7 {
	return &IndexedCompressedChain7{indexedCore: buildIndexedCore[[7]TokenID](corpus)}
}

func BuildCompressedIndexed8(corpus [][]TokenID) *IndexedCompressedChain8 {
	return &IndexedCompressedChain8{indexedCore: buildIndexedCore[[8]TokenID](corpus)}
}
