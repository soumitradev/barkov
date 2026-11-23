package interned

// LookupStatus describes why a single context order was used or skipped in moveTail.
type LookupStatus string

const (
	LookupUsed                  LookupStatus = "used"
	LookupNoCorpusMatch         LookupStatus = "no_corpus_match"
	LookupBelowMinSupport       LookupStatus = "below_min_support"
	LookupAllOptionsCopyBlocked LookupStatus = "all_options_copy_blocked"
	LookupHigherOrderUsed       LookupStatus = "higher_order_used"
)

// LookupRecord captures one order's evaluation inside backoffCore.moveTail.
type LookupRecord struct {
	Order     int
	Status    LookupStatus
	SeenCount uint32
}

// StepRecord captures one live sampling decision made by backoffCore.moveTail.
type StepRecord struct {
	UsedOrder int
	Picked    TokenID
	End       bool
	Lookups   []LookupRecord
}

// StepObserver receives live step decisions when attached via WithObserver.
type StepObserver func(StepRecord)

// WithObserver returns a view sharing the same tables that reports every live
// step decision to fn. Nil disables observation with zero overhead.
func (b *backoffCore) WithObserver(fn StepObserver) BackoffChain {
	view := *b
	view.observer = fn
	return wrapBackoff(&view)
}

// observeStep reports a step that sampled tok at order used. The shorter
// orders were never consulted, so they are recorded as skipped with their
// observation counts.
func (b *backoffCore) observeStep(key contextKey, used int, obs uint32, tok TokenID, lookups []LookupRecord) {
	lookups = append(lookups, LookupRecord{Order: used, Status: LookupUsed, SeenCount: obs})
	for m := used - 1; m >= b.cfg.MinOrder; m-- {
		var seen uint32
		if idx, ok := b.tables[m-1].lookup(key); ok {
			seen = idx.Obs
		}
		lookups = append(lookups, LookupRecord{Order: m, Status: LookupHigherOrderUsed, SeenCount: seen})
	}
	b.observer(StepRecord{UsedOrder: used, Picked: tok, End: tok == b.sentinels.End, Lookups: lookups})
}
