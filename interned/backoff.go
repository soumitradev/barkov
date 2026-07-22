package interned

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"sync"
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
// skipped in favor of shorter contexts with more evidence. When steering
// is active and every candidate at an order would extend a verbatim
// corpus run, the walk falls through to the next shorter order, whose
// candidate set is wider; if every order is exhausted the step fails
// with ErrSentenceFailedValidation and the caller retries the sentence.
func (b *backoffCore) moveTail(window []TokenID) (TokenID, error) {
	// One steering snapshot per step, shared by every candidate check at
	// every order: all of them test membership in the same follower set.
	var sv steerView
	if b.cfg.MaxVerbatim > 0 && b.runLongEnough(window) {
		sv = b.steerSnapshot(window)
	}
	// Steering-probe buffer for the steerBig fallback, shared across every
	// candidate probe below. Declaring it inside wouldExtendVerbatim would
	// heap-allocate per candidate: the buffer's address flows through the
	// orderTable interface call, so escape analysis can't stack it, and
	// order-1 contexts on common tokens have thousands of candidates.
	var probeBuf [8]TokenID
	sawState := false
	for m := b.cfg.MaxOrder; m >= 1; m-- {
		tail := window[len(window)-m:]
		idx, ok := b.tables[m-1].lookupTail(tail)
		if !ok {
			continue
		}
		if m > 1 && uint32(idx.Obs) < uint32(b.cfg.MinSupport) {
			continue // thin evidence; a shorter order has more
		}
		sawState = true
		if tok, ok := b.sample(m, idx, window, sv, &probeBuf); ok {
			return tok, nil
		}
		// All candidates at this order extend a verbatim run; back off.
	}
	var zero TokenID
	if sawState {
		return zero, fmt.Errorf("barkov: all continuations extend a verbatim run: %w",
			barkov.ErrSentenceFailedValidation)
	}
	return zero, fmt.Errorf("barkov: state %v not in model at any order: %w",
		window, barkov.ErrStateNotFound)
}

// runLongEnough reports whether the window's last MaxVerbatim-1 tokens
// are all real (no Begin sentinel): only then can emitting a candidate
// complete a MaxVerbatim-length run of real tokens worth filtering.
// Matches the per-window validator, which also only sees full windows.
func (b *backoffCore) runLongEnough(window []TokenID) bool {
	for _, tok := range window[len(window)-(b.cfg.MaxVerbatim-1):] {
		if tok == b.sentinels.Begin {
			return false
		}
	}
	return true
}

// steerMode classifies the per-step steering snapshot.
type steerMode uint8

const (
	steerOff     steerMode = iota // steering inactive (or nothing to filter)
	steerSingle                   // steering state has exactly one follower
	steerSet                      // small follower set: linear membership scan
	steerBig                      // large follower set: per-candidate table probe
)

// steerView is the per-step snapshot of the steering state: the followers
// of the window's last MaxVerbatim-1 tokens in the order-(MaxVerbatim-1)
// table. A candidate completes a verbatim corpus MaxVerbatim-gram iff it
// belongs to that follower set — the (MaxVerbatim-1)-gram's followers are
// exactly the tokens that complete a corpus MaxVerbatim-gram — so one
// lookup per step replaces the per-candidate probes of the larger
// order-MaxVerbatim table that steering used to cost.
type steerView struct {
	mode   steerMode
	single TokenID   // steerSingle: the sole follower
	set    []TokenID // steerSet: follower slice (aliases chain storage; read-only)
}

// steerProbeFallback caps the follower-set size handled by membership
// scan; larger sets fall back to one order-MaxVerbatim table probe per
// candidate (the pre-snapshot behavior). On prose corpora ~98% of
// steering states have a single follower and >99% have at most two, so
// the fallback is reached only from a handful of high-fanout states.
const steerProbeFallback = 16

// steerSnapshot resolves the steering state for window once: the
// order-(r-1) table entry for the window's last r-1 tokens, where
// r = MaxVerbatim. The zero value (steerOff) means no candidate can
// complete a verbatim gram at this step.
func (b *backoffCore) steerSnapshot(window []TokenID) steerView {
	r := b.cfg.MaxVerbatim
	idx, ok := b.tables[r-2].lookupTail(window[len(window)-(r-1):])
	if !ok {
		return steerView{}
	}
	if idx.Count == 1 {
		return steerView{mode: steerSingle, single: TokenID(idx.Offset)}
	}
	set, _ := b.tables[r-2].slices(idx)
	if len(set) > steerProbeFallback {
		return steerView{mode: steerBig}
	}
	return steerView{mode: steerSet, set: set}
}

// filtered reports whether emitting tok after window would complete a
// verbatim MaxVerbatim-gram. End is never filtered: no order-MaxVerbatim
// state contains End (windows stop before it), so the original
// per-candidate probe always missed for End — but End IS present in
// follower sets as the terminal transition, so it must be excluded
// explicitly here to preserve those semantics (and let sentences
// terminate under maximal steering).
func (b *backoffCore) filtered(sv steerView, window []TokenID, tok TokenID, probeBuf *[8]TokenID) bool {
	if tok == b.sentinels.End {
		return false
	}
	switch sv.mode {
	case steerSingle:
		return tok == sv.single
	case steerSet:
		for _, f := range sv.set {
			if f == tok {
				return true
			}
		}
		return false
	case steerBig:
		return b.wouldExtendVerbatim(window, tok, probeBuf)
	}
	return false
}

// wouldExtendVerbatim reports whether emitting tok after window would
// complete a MaxVerbatim-gram that appears verbatim in the corpus. The
// order-R table doubles as the R-gram existence set: a context state
// exists iff that R-gram occurs inside a message. END can never be
// filtered: no state contains END, so the probe (.., END) always misses
// and sentences can always terminate. probe is the caller's per-step
// scratch buffer (see moveTail). Used only on the steerBig fallback path;
// the common path tests follower-set membership via filtered instead.
func (b *backoffCore) wouldExtendVerbatim(window []TokenID, tok TokenID, probe *[8]TokenID) bool {
	r := b.cfg.MaxVerbatim
	copy(probe[:r-1], window[len(window)-(r-1):])
	probe[r-1] = tok
	_, exists := b.tables[r-1].lookupTail(probe[:r])
	return exists
}

// sample draws a follower from the order-m state idx, reproducing
// pickFollow semantics: Count==1 states pack the follower into Offset;
// otherwise a weighted draw over the cumulative distribution. With
// steering active, candidates that would extend a verbatim run are
// excluded and the survivors keep their original weights; ok=false
// means every candidate was filtered.
func (b *backoffCore) sample(m int, idx barkov.ChoicesIndex, window []TokenID, sv steerView, probeBuf *[8]TokenID) (TokenID, bool) {
	if idx.Count == 1 {
		tok := TokenID(idx.Offset)
		if sv.mode != steerOff && b.filtered(sv, window, tok, probeBuf) {
			return 0, false
		}
		return tok, true
	}
	choices, cum := b.tables[m-1].slices(idx)
	if sv.mode == steerOff {
		return choices[b.draw(cum)], true
	}
	if sv.mode == steerBig {
		// Membership test is a table probe here, so keep the weighted
		// reservoir (one probe per candidate, no second pass): each
		// survivor replaces the pick with probability weight/totalSoFar,
		// an exact weighted draw over the filtered set.
		var pick TokenID
		var total uint32
		var prev uint32
		for i, tok := range choices {
			w := cum[i] - prev
			prev = cum[i]
			if b.filtered(sv, window, tok, probeBuf) {
				continue
			}
			total += w
			if b.uint32n(total) < w {
				pick = tok
			}
		}
		if total == 0 {
			return 0, false
		}
		return pick, true
	}
	// Membership test is a cheap compare/scan, so do a two-pass weighted
	// draw: total the survivors' weights, then draw once and walk. Same
	// distribution as a draw over the filtered set, but one RNG call
	// instead of one per survivor — order-1 contexts can have thousands
	// of candidates.
	var total uint32
	var prev uint32
	for i := range choices {
		w := cum[i] - prev
		prev = cum[i]
		if !b.filtered(sv, window, choices[i], probeBuf) {
			total += w
		}
	}
	if total == 0 {
		return 0, false
	}
	choiceNum := b.uint32n(total)
	prev = 0
	var acc uint32
	for i := range choices {
		w := cum[i] - prev
		prev = cum[i]
		if b.filtered(sv, window, choices[i], probeBuf) {
			continue
		}
		acc += w
		if acc > choiceNum {
			return choices[i], true
		}
	}
	return 0, false // unreachable: acc climbs to total > choiceNum
}

// uint32n draws from [0, n) using the configured RNG.
func (b *backoffCore) uint32n(n uint32) uint32 {
	if b.rng != nil {
		return b.rng.Uint32N(n)
	}
	return rand.Uint32N(n)
}

// draw picks an index into a cumulative distribution by weight.
func (b *backoffCore) draw(cum []uint32) int {
	choiceNum := b.uint32n(cum[len(cum)-1])
	return scanCumDist(cum, choiceNum)
}

// scanCumDist returns the first index whose cumulative weight exceeds
// choiceNum. Fanout averages ~1.1-1.8, so a linear scan wins for small
// groups; begin-state fanouts in the thousands flip the crossover to
// binary search.
func scanCumDist(cum []uint32, choiceNum uint32) int {
	if len(cum) <= 16 {
		for i, c := range cum {
			if c > choiceNum {
				return i
			}
		}
		return len(cum) - 1 // unreachable: choiceNum < cum[len-1]
	}
	return sort.Search(len(cum), func(i int) bool { return cum[i] > choiceNum })
}

// buildBackoffCore builds one indexedCore per order 1..MaxOrder, one
// corpus pass each. The per-order builds are independent (corpus is
// read-only, each goroutine writes its own table slot), so they run
// concurrently — a MaxOrder-6 build is six full passes, and wall time
// drops to roughly the slowest single pass on multicore machines.
func buildBackoffCore(cfg BackoffConfig, corpus [][]TokenID) *backoffCore {
	tables := make([]orderTable, cfg.MaxOrder)
	var wg sync.WaitGroup
	for m := 1; m <= cfg.MaxOrder; m++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tables[m-1] = buildOrderTable(m, corpus)
		}()
	}
	wg.Wait()
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
