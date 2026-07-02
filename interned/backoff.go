package interned

import (
	"fmt"
	"math/rand/v2"
	"unsafe"

	barkov "github.com/soumitradev/barkov/v2"
)

// BackoffConfig configures BuildBackoff. The zero value gives defaults.
type BackoffConfig struct {
	// MaxOrder is the longest context tried, 2..8. Default 6.
	MaxOrder int
	// MinSupport is the minimum observation count (ChoicesIndex.Obs) a
	// context at order >= 2 needs to be used; order 1 is exempt (it is
	// the unconditional fallback). Default 3. Range 1..65535.
	MinSupport int
	// MaxVerbatim is the longest run of consecutive emitted tokens
	// allowed to reproduce a corpus n-gram verbatim. Steering filters
	// candidates that would extend a run to this length. 0 = default
	// (MaxOrder); -1 = disabled; else 2..MaxOrder.
	MaxVerbatim int
}

// withDefaults validates cfg and fills zero values with defaults. Panics
// on out-of-range values, matching the Build convention.
func (cfg BackoffConfig) withDefaults() BackoffConfig {
	if cfg.MaxOrder == 0 {
		cfg.MaxOrder = 6
	}
	if cfg.MaxOrder < 2 || cfg.MaxOrder > 8 {
		panic(fmt.Sprintf("interned: BuildBackoff: MaxOrder %d outside supported range 2..8", cfg.MaxOrder))
	}
	if cfg.MinSupport == 0 {
		cfg.MinSupport = 3
	}
	if cfg.MinSupport < 1 || cfg.MinSupport > 65535 {
		panic(fmt.Sprintf("interned: BuildBackoff: MinSupport %d outside supported range 1..65535", cfg.MinSupport))
	}
	if cfg.MaxVerbatim == 0 {
		cfg.MaxVerbatim = cfg.MaxOrder
	}
	if cfg.MaxVerbatim != -1 && (cfg.MaxVerbatim < 2 || cfg.MaxVerbatim > cfg.MaxOrder) {
		panic(fmt.Sprintf("interned: BuildBackoff: MaxVerbatim %d must be -1 (disabled) or in 2..MaxOrder (%d)", cfg.MaxVerbatim, cfg.MaxOrder))
	}
	return cfg
}

// orderTable is the view backoffCore needs of one per-order indexedCore.
type orderTable interface {
	// lookupTail probes the table with the last len(tail) tokens of the
	// window. Caller guarantees len(tail) == the table's stateSize; the
	// implementation reinterprets tail's backing memory as K, same idiom
	// as indexedCore.MoveKey.
	lookupTail(tail []TokenID) (barkov.ChoicesIndex, bool)
	// slices resolves a Count>1 index to its choices/cumdist views.
	slices(idx barkov.ChoicesIndex) (choices []TokenID, cum []uint32)
}

// lookupTail implements orderTable. tail's backing array has at least
// stateSize tokens, so reading sizeof(K) bytes from &tail[0] produces the
// intended [stateSize]TokenID key value.
func (c *indexedCore[K]) lookupTail(tail []TokenID) (barkov.ChoicesIndex, bool) {
	return c.Model.Get(*(*K)(unsafe.Pointer(&tail[0])))
}

// slices implements orderTable.
func (c *indexedCore[K]) slices(idx barkov.ChoicesIndex) ([]TokenID, []uint32) {
	return c.Choices[idx.Offset : idx.Offset+uint32(idx.Count)],
		c.CumDist[idx.Offset : idx.Offset+uint32(idx.Count)]
}

// backoffCore does the variable-order walk over a stack of per-order
// indexed tables. At each step it tries the longest context first and
// backs off to shorter ones until it finds a state with enough corpus
// support (MinSupport); order 1 is the unconditional fallback. The walk
// is stateless — a pure function of the state window plus RNG — so a
// built chain is safe for concurrent readers, except under SetRNG.
type backoffCore struct {
	tables    []orderTable // index m-1 = order m
	cfg       BackoffConfig
	sentinels barkov.Sentinels[TokenID]
	encoder   PackedEncoder
	rng       *rand.Rand
}

// SetRNG overrides the random source used by the walk. Intended for
// deterministic tests and benchmarks; not safe for concurrent use.
func (b *backoffCore) SetRNG(r *rand.Rand) { b.rng = r }

func (b *backoffCore) StateSize() int                        { return b.cfg.MaxOrder }
func (b *backoffCore) Sentinels() barkov.Sentinels[TokenID]  { return b.sentinels }
func (b *backoffCore) Encoder() barkov.StateEncoder[TokenID] { return b.encoder }

// Move satisfies barkov.GenerativeChain[TokenID]. Same reinterpret idiom
// as indexedCore.Move: the packed-encoder byte layout matches TokenID's
// in-memory layout, so the string bytes are viewed as a token window
// without a decode loop.
func (b *backoffCore) Move(state string) (TokenID, error) {
	if len(state) != b.cfg.MaxOrder*4 {
		return 0, fmt.Errorf("barkov: state %q not in model: %w", state, barkov.ErrStateNotFound)
	}
	window := unsafe.Slice((*TokenID)(unsafe.Pointer(unsafe.StringData(state))), b.cfg.MaxOrder)
	return b.moveTail(window)
}

// moveTail picks the next token for the given window (length MaxOrder):
// longest supported context wins; orders with Obs below MinSupport are
// skipped in favor of shorter contexts with more evidence.
func (b *backoffCore) moveTail(window []TokenID) (TokenID, error) {
	for m := b.cfg.MaxOrder; m >= 1; m-- {
		tail := window[len(window)-m:]
		idx, ok := b.tables[m-1].lookupTail(tail)
		if !ok {
			continue
		}
		if m > 1 && uint32(idx.Obs) < uint32(b.cfg.MinSupport) {
			continue // thin evidence; a shorter order has more
		}
		return b.sample(m, idx)
	}
	var zero TokenID
	return zero, fmt.Errorf("barkov: state %v not in model at any order: %w",
		window, barkov.ErrStateNotFound)
}

// sample draws a follower from the order-m state idx, reproducing
// pickFollow semantics: Count==1 states pack the follower into Offset;
// otherwise a weighted draw over the cumulative distribution.
func (b *backoffCore) sample(m int, idx barkov.ChoicesIndex) (TokenID, error) {
	if idx.Count == 1 {
		return TokenID(idx.Offset), nil
	}
	choices, cum := b.tables[m-1].slices(idx)
	return choices[b.draw(cum)], nil
}

// draw picks an index into a cumulative distribution by weight.
func (b *backoffCore) draw(cum []uint32) int {
	var choiceNum uint32
	if b.rng != nil {
		choiceNum = b.rng.Uint32N(cum[len(cum)-1])
	} else {
		choiceNum = rand.Uint32N(cum[len(cum)-1])
	}
	// Fanout averages ~1.1-1.8; linear scan beats sort.Search at that size.
	for i, c := range cum {
		if c > choiceNum {
			return i
		}
	}
	return len(cum) - 1 // unreachable: choiceNum < cum[len-1]
}

// buildBackoffCore builds one indexedCore per order 1..MaxOrder, one
// corpus pass each. Build is a rare event (corpus refreshes); fusing the
// passes is deliberately out of scope.
func buildBackoffCore(cfg BackoffConfig, corpus [][]TokenID) *backoffCore {
	tables := make([]orderTable, cfg.MaxOrder)
	for m := 1; m <= cfg.MaxOrder; m++ {
		tables[m-1] = buildOrderTable(m, corpus)
	}
	return &backoffCore{
		tables:    tables,
		cfg:       cfg,
		sentinels: DefaultSentinels(),
		encoder:   PackedEncoder{},
	}
}

// buildOrderTable monomorphizes buildIndexedCore for one order.
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

// BuildBackoff constructs a variable-order chain from a pre-interned
// corpus: at each step the walk uses the longest context (up to
// cfg.MaxOrder) with at least cfg.MinSupport corpus observations, backing
// off to shorter contexts — and ultimately order 1 — where the data is
// thin. Compared to a fixed-order chain, which replays corpus spans
// verbatim wherever a long context has exactly one continuation, backoff
// samples where the corpus genuinely branches.
//
// The zero-value config gives MaxOrder 6, MinSupport 3, and verbatim
// steering at window MaxOrder. Panics on out-of-range config values,
// matching the Build convention.
//
// The returned value implements barkov.GenerativeChain[TokenID] with
// StateSize() == MaxOrder; Gen/GenIter work unchanged. Deterministic
// tests reach the RNG through barkov.RNGSettable:
//
//	chain := interned.BuildBackoff(interned.BackoffConfig{}, encoded)
//	chain.(barkov.RNGSettable).SetRNG(r)
func BuildBackoff(cfg BackoffConfig, corpus [][]TokenID) barkov.GenerativeChain[TokenID] {
	cfg = cfg.withDefaults()
	core := buildBackoffCore(cfg, corpus)
	switch cfg.MaxOrder {
	case 2:
		return &backoffChain2{core}
	case 3:
		return &backoffChain3{core}
	case 4:
		return &backoffChain4{core}
	case 5:
		return &backoffChain5{core}
	case 6:
		return &backoffChain6{core}
	case 7:
		return &backoffChain7{core}
	case 8:
		return &backoffChain8{core}
	default:
		panic("unreachable: withDefaults validated MaxOrder")
	}
}
