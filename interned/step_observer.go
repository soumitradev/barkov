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
