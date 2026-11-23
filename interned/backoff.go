package interned

import (
	"fmt"
	"iter"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

// BackoffChain is the variable-order chain BuildBackoff returns.
type BackoffChain interface {
	barkov.GenerativeChain[TokenID]
	barkov.RNGSettable
	// States iterates the states at order 1..MaxOrder with their observation counts.
	States(order int) iter.Seq2[[]TokenID, uint32]
	// SeedContext samples observed history for a short interior seed.
	SeedContext(seed []TokenID) ([]TokenID, error)
	// WithoutVerbatim returns a view of the same tables with copy steering off.
	WithoutVerbatim() BackoffChain
	// CopiesRun reports whether tokens contain a corpus run that copy steering
	// would have prevented inside one walk. False when steering is disabled.
	CopiesRun(tokens []TokenID) bool
	// IsFragment reports whether tokens occur contiguously inside one corpus
	// message. Only runs of up to MaxOrder tokens can match.
	IsFragment(tokens []TokenID) bool
	// WithEscape returns a view of the same tables whose walk interpolates
	// across context orders. scale 0 restores the longest-context walk.
	WithEscape(scale float64) BackoffChain
	// Options explains one step of the walk after window.
	Options(window []TokenID) ([]OrderOptions, int)
}

// contextKey holds up to 8 context tokens right-aligned, so the order-m
// table reads its last m tokens and one key serves every order. Passing it
// by value through orderTable keeps it on the stack, where a slice would escape.
type contextKey [8]TokenID

// keyOf right-aligns up to 8 tokens into a contextKey
func keyOf(tokens []TokenID) contextKey {
	var key contextKey
	copy(key[len(key)-len(tokens):], tokens)
	return key
}

// shift drops the oldest token and appends tok
func (key contextKey) shift(tok TokenID) contextKey {
	copy(key[:], key[1:])
	key[len(key)-1] = tok
	return key
}

// orderTable is one per-order indexedCore, whatever its key size
type orderTable interface {
	// lookup reads the table's stateSize tokens at the end of key
	lookup(key contextKey) (barkov.ChoicesIndex, bool)
	// slices resolves a Count>1 index to its followers and cumulative weights
	slices(idx barkov.ChoicesIndex) (choices []TokenID, cum []uint32)
	states() iter.Seq2[[]TokenID, uint32]
	compact()
	writeTable(e *snapshotEncoder)
	snapshotStateCount() int
}

func (c *indexedCore[K]) lookup(key contextKey) (barkov.ChoicesIndex, bool) {
	return c.Model.Get(*(*K)(unsafe.Pointer(&key[len(key)-c.stateSize])))
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

// backoffCore walks per-order tables from the longest context down to the
// first with MinSupport observations. Concurrent readers are safe unless
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

// moveTail picks the next token after window. When steering vetoes every
// candidate at an order, the next shorter order offers a wider set.
func (b *backoffCore) moveTail(window []TokenID) (TokenID, error) {
	key := keyOf(window[len(window)-b.cfg.MaxOrder:])
	var sv steerView
	if b.cfg.MaxVerbatim > 0 && b.runLongEnough(key) {
		sv = b.steerSnapshot(key)
	}
	if b.escape > 0 {
		return b.moveEscape(key, sv)
	}
	failure := noContext
	for m := b.cfg.MaxOrder; m >= b.cfg.MinOrder; m-- {
		idx, ok := b.tables[m-1].lookup(key)
		if !ok {
			continue
		}
		failure = max(failure, thinContext)
		if idx.Obs < uint32(b.cfg.MinSupport) {
			continue
		}
		failure = allSteered
		if tok, ok := b.sample(m, idx, key, sv); ok {
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

// errAllSteered is built once, since steered walks dead-end often enough
// that formatting it per failed step shows up in allocation profiles
var errAllSteered = fmt.Errorf("barkov: all continuations extend a verbatim run: %w", barkov.ErrSentenceFailedValidation)

func (b *backoffCore) noContinuationErr(failure stepFailure) error {
	switch failure {
	case allSteered:
		return errAllSteered
	case thinContext:
		return fmt.Errorf("barkov: observed contexts at orders %d..%d have fewer than %d observations: %w",
			b.cfg.MinOrder, b.cfg.MaxOrder, b.cfg.MinSupport, barkov.ErrInsufficientSupport)
	default:
		return fmt.Errorf("barkov: no observed context at orders %d..%d: %w",
			b.cfg.MinOrder, b.cfg.MaxOrder, barkov.ErrStateNotFound)
	}
}

// IsFragment implements BackoffChain. Every run inside a message is the
// context before its next token, so it is a key of the table of its length.
func (b *backoffCore) IsFragment(tokens []TokenID) bool {
	if len(tokens) == 0 || len(tokens) > b.cfg.MaxOrder {
		return false
	}
	_, ok := b.tables[len(tokens)-1].lookup(keyOf(tokens))
	return ok
}
