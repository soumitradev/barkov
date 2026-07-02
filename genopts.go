package barkov

import "runtime"

// StuckDetector is the interface for stuck-state detection. Implementations
// track which seeds repeatedly fail and allow short-circuiting retries.
// The barkov/stuck subpackage provides the default implementation.
type StuckDetector interface {
	RecordFailure(state string) bool // returns true if now considered stuck
	RecordSuccess(state string)
	IsStuck(state string) bool
}

// SlicePool is an optional pool for reusing slice allocations across
// generation attempts. The core package ships with NoPool (no reuse).
// The barkov/interned subpackage provides a sync.Pool-backed implementation.
type SlicePool[T any] interface {
	GetState() *[]T
	PutState(*[]T)
	GetGenerated() *[]T
	PutGenerated(*[]T)
}

// NoPool is a SlicePool that allocates fresh slices every time.
type NoPool[T any] struct{}

func (NoPool[T]) GetState() *[]T        { s := make([]T, 0, 16); return &s }
func (NoPool[T]) PutState(*[]T)         {}
func (NoPool[T]) GetGenerated() *[]T    { s := make([]T, 0, 128); return &s }
func (NoPool[T]) PutGenerated(*[]T)     {}

// GenOption configures a generation call. All options are functional;
// the unexported config struct keeps internals private.
type GenOption[T comparable] func(*genConfig[T])

type genConfig[T comparable] struct {
	seed            []T
	validator       func([]T) bool
	validatorWidth  int // window width for validator; 0 = default to StateSize+2
	parallelism     int // 0 = single-threaded; >0 = goroutine count
	stuckCache      StuckDetector
	pool            SlicePool[T]
	outputValidator func([]T) bool
}

// WithSeed prepends seed tokens to the generated sequence.
// Padding with Sentinels.Begin happens automatically if seed is shorter
// than chain.StateSize().
func WithSeed[T comparable](seed []T) GenOption[T] {
	return func(c *genConfig[T]) { c.seed = seed }
}

// WithValidator installs an n-gram validator. The function is called
// with the last StateSize+2 tokens after each step; returning false aborts
// the current attempt with ErrSentenceFailedValidation.
//
// The window width is fixed at StateSize+2. Outputs shorter than that
// are never passed to the validator; use WithOutputValidator to check
// short or complete outputs.
func WithValidator[T comparable](v func([]T) bool) GenOption[T] {
	return func(c *genConfig[T]) { c.validator = v }
}

// WithOutputValidator installs a whole-output validator. It is called
// exactly once with the complete output (seed tokens plus generated,
// sentinels excluded — the same slice Gen would return); returning false
// aborts the attempt with ErrSentenceFailedValidation. Unlike
// WithValidator, this also covers outputs shorter than StateSize+2.
//
// An output validator makes generation non-streaming by definition: the
// single-threaded path must buffer the entire output before it can call
// v, then replays it through the iterator.
func WithOutputValidator[T comparable](v func([]T) bool) GenOption[T] {
	return func(c *genConfig[T]) { c.outputValidator = v }
}

// WithNGramValidator installs a WindowValidator — a per-window validator
// that knows its own width. Gen slides a window of exactly v.N() tokens
// and calls v.Validate on it, eliminating the silent width mismatch that
// WithValidator has when hand-wired to an NGramSet built with a
// different n. T is inferred from v.
func WithNGramValidator[T comparable](v WindowValidator[T]) GenOption[T] {
	return func(c *genConfig[T]) {
		c.validator = v.Validate
		c.validatorWidth = v.N()
	}
}

// WithThreaded fans the generation attempt out across runtime.NumCPU()*8
// goroutines and returns the first successful result.
func WithThreaded[T comparable]() GenOption[T] {
	return func(c *genConfig[T]) { c.parallelism = runtime.NumCPU() * 8 }
}

// WithParallelism overrides the threaded fan-out width. n=1 is
// equivalent to the unthreaded path. n=0 (the default) means single-threaded.
func WithParallelism[T comparable](n int) GenOption[T] {
	return func(c *genConfig[T]) { c.parallelism = n }
}

// WithStuckDetector installs a stuck-state cache. Seeds that fail
// repeatedly will short-circuit instead of being retried. Only effective
// when used with WithThreaded or WithParallelism(>1).
func WithStuckDetector[T comparable](d StuckDetector) GenOption[T] {
	return func(c *genConfig[T]) { c.stuckCache = d }
}

// WithSlicePool overrides the default slice allocation strategy for
// the duration of this call.
func WithSlicePool[T comparable](p SlicePool[T]) GenOption[T] {
	return func(c *genConfig[T]) { c.pool = p }
}
