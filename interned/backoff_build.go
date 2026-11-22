package interned

import (
	"fmt"
	"sync"
)

// BuildBackoff constructs a variable-order chain from a pre-interned
// corpus. Each step uses the longest context (up to cfg.MaxOrder) with at
// least cfg.MinSupport observations, backing off down to MinOrder, so it
// samples where the corpus branches instead of replaying spans verbatim
// like a fixed-order chain. StateSize() is MaxOrder. Panics on
// out-of-range config values.
func BuildBackoff(cfg BackoffConfig, corpus [][]TokenID) BackoffChain {
	cfg = cfg.withDefaults()
	if err := cfg.validate(); err != nil {
		panic(fmt.Sprintf("interned: BuildBackoff: %v", err))
	}
	return wrapBackoff(buildBackoffCore(cfg, corpus))
}

// buildBackoffCore builds the per-order tables concurrently, since each is
// an independent pass over the read-only corpus
func buildBackoffCore(cfg BackoffConfig, corpus [][]TokenID) *backoffCore {
	tables := make([]orderTable, cfg.MaxOrder)
	var wg sync.WaitGroup
	for m := 1; m <= cfg.MaxOrder; m++ {
		wg.Go(func() { tables[m-1] = buildOrderTable(m, corpus) })
	}
	wg.Wait()
	var contexts *contextIndex
	if cfg.MinOrder > 1 {
		contexts = newContextIndex(tables[cfg.MinOrder-1], cfg.MinOrder, cfg.MinSupport)
	}
	return &backoffCore{
		tables:    tables,
		contexts:  contexts,
		cfg:       cfg,
		sentinels: DefaultSentinels(),
		encoder:   PackedEncoder{},
	}
}

func buildOrderTable(order int, corpus [][]TokenID) orderTable {
	switch order {
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
		panic(fmt.Sprintf("interned: order %d outside supported range 1..8", order))
	}
}
