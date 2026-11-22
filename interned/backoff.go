package interned

import (
	"fmt"
	"iter"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

// BackoffChain is a backoff engine chain that supports state iteration
// across its orders and RNG configuration.
type BackoffChain interface {
	barkov.GenerativeChain[TokenID]
	barkov.RNGSettable
	// States iterates all states at order (1..MaxOrder) with their total
	// transition observation counts.
	States(order int) iter.Seq2[[]TokenID, uint32]
	// SeedContext samples observed history for a short interior seed.
	SeedContext(seed []TokenID) ([]TokenID, error)
	// WithoutVerbatim returns a view sharing the same immutable model tables.
	WithoutVerbatim() BackoffChain
	// CopiesRun reports whether tokens contain a corpus n-gram as long as
	// the copy-steering window, i.e. a run that steering would have
	// prevented inside one walk. False when steering is disabled.
	CopiesRun(tokens []TokenID) bool
	// IsFragment reports whether tokens occur contiguously inside one
	// corpus message. Exact for up to MaxOrder tokens; longer runs are
	// what copy steering prevents.
	IsFragment(tokens []TokenID) bool
	// WithEscape returns a view sharing the same tables whose walk
	// interpolates across context orders instead of always using the
	// longest one. scale 0 restores the default.
	WithEscape(scale float64) BackoffChain
	// Options explains one step of the walk after window.
	Options(window []TokenID) ([]OrderOptions, int)
}

// orderTable is the view backoffCore needs of one per-order indexedCore
type orderTable interface {
	// lookupTail probes with tail, which must hold exactly the table's
	// stateSize tokens
	lookupTail(tail []TokenID) (barkov.ChoicesIndex, bool)
	// slices resolves a Count>1 index to its followers and cumulative weights
	slices(idx barkov.ChoicesIndex) (choices []TokenID, cum []uint32)
	states() iter.Seq2[[]TokenID, uint32]
}

func (c *indexedCore[K]) lookupTail(tail []TokenID) (barkov.ChoicesIndex, bool) {
	return c.Model.Get(*(*K)(unsafe.Pointer(&tail[0])))
}

func (c *indexedCore[K]) slices(idx barkov.ChoicesIndex) ([]TokenID, []uint32) {
	return c.Choices[idx.Offset : idx.Offset+idx.Count],
		c.CumDist[idx.Offset : idx.Offset+idx.Count]
}

func (c *indexedCore[K]) states() iter.Seq2[[]TokenID, uint32] {
	return func(yield func([]TokenID, uint32) bool) {
		for k, idx := range c.Model.All() {
			var total uint32
			if idx.Count == 1 {
				total = idx.Obs
			} else {
				total = c.CumDist[idx.Offset+idx.Count-1]
			}
			tokens := make([]TokenID, c.stateSize)
			copy(tokens, unsafe.Slice((*TokenID)(unsafe.Pointer(&k)), c.stateSize))
			if !yield(tokens, total) {
				return
			}
		}
	}
}

// backoffCore walks a stack of per-order tables, trying the longest context
// first and backing off until one has MinSupport observations. The walk
// depends only on the window and RNG, so concurrent readers are safe unless
// SetRNG is used.
type backoffCore struct {
	random
	tables    []orderTable // index m-1 = order m
	cfg       BackoffConfig
	sentinels barkov.Sentinels[TokenID]
	encoder   PackedEncoder
	contexts  *contextIndex
	escape    float64 // 0 means the longest supported context always wins
}

func (b *backoffCore) States(order int) iter.Seq2[[]TokenID, uint32] {
	if order < 1 || order > len(b.tables) {
		return func(yield func([]TokenID, uint32) bool) {}
	}
	return b.tables[order-1].states()
}

func (b *backoffCore) StateSize() int                        { return b.cfg.MaxOrder }
func (b *backoffCore) Sentinels() barkov.Sentinels[TokenID]  { return b.sentinels }
func (b *backoffCore) Encoder() barkov.StateEncoder[TokenID] { return b.encoder }

func (b *backoffCore) Move(state string) (TokenID, error) {
	if len(state) != b.cfg.MaxOrder*4 {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	window := unsafe.Slice((*TokenID)(unsafe.Pointer(unsafe.StringData(state))), b.cfg.MaxOrder)
	return b.moveTail(window)
}

// moveTail picks the next token after window (length MaxOrder). When
// steering vetoes every candidate at an order the walk falls through to the
// next shorter one, whose candidate set is wider.
func (b *backoffCore) moveTail(window []TokenID) (TokenID, error) {
	var sv steerView
	if b.cfg.MaxVerbatim > 0 && b.runLongEnough(window) {
		sv = b.steerSnapshot(window)
	}
	// Declared here because escape analysis can't keep it on the stack once
	// its address passes through the orderTable interface, and order-1
	// contexts can have thousands of candidates
	var probeBuf [8]TokenID
	if b.escape > 0 {
		return b.moveEscape(window, sv, &probeBuf)
	}
	failure := noContext
	for m := b.cfg.MaxOrder; m >= b.cfg.MinOrder; m-- {
		idx, ok := b.tables[m-1].lookupTail(window[len(window)-m:])
		if !ok {
			continue
		}
		failure = max(failure, thinContext)
		if idx.Obs < uint32(b.cfg.MinSupport) {
			continue
		}
		failure = allSteered
		if tok, ok := b.sample(m, idx, window, sv, &probeBuf); ok {
			return tok, nil
		}
	}
	return 0, b.noContinuationErr(failure)
}

// stepFailure is how far a step got before running out of continuations
type stepFailure uint8

const (
	noContext   stepFailure = iota // no order has seen the window's context
	thinContext                    // every seen context is below MinSupport
	allSteered                     // steering vetoed every supported follower
)

func (b *backoffCore) noContinuationErr(failure stepFailure) error {
	switch failure {
	case allSteered:
		return fmt.Errorf("barkov: all continuations extend a verbatim run: %w", barkov.ErrSentenceFailedValidation)
	case thinContext:
		return fmt.Errorf("barkov: observed contexts at orders %d..%d have fewer than %d observations: %w",
			b.cfg.MinOrder, b.cfg.MaxOrder, b.cfg.MinSupport, barkov.ErrInsufficientSupport)
	default:
		return fmt.Errorf("barkov: no observed context at orders %d..%d: %w",
			b.cfg.MinOrder, b.cfg.MaxOrder, barkov.ErrStateNotFound)
	}
}

// IsFragment implements BackoffChain. Every run of up to MaxOrder tokens
// inside a message is the context before its next token (or END), so the
// run is a key of the table of its own length.
func (b *backoffCore) IsFragment(tokens []TokenID) bool {
	if len(tokens) == 0 || len(tokens) > b.cfg.MaxOrder {
		return false
	}
	_, ok := b.tables[len(tokens)-1].lookupTail(tokens)
	return ok
}
