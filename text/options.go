package text

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"time"
)

// avDefault is the sentinel config value meaning "anti-verbatim window not
// set; default it to Order()+2". A user-supplied 0 disables anti-verbatim;
// a positive value fixes the window.
const avDefault = -1

type config struct {
	generationConfig
	order         int
	minOrder      int
	minSupport    int
	antiVerbatim  int
	bidirectional bool
	rng           *rand.Rand
}

func defaultConfig() config {
	return config{
		generationConfig: generationConfig{timeout: 10 * time.Second, retries: 64},
		order:            4, minOrder: 1, minSupport: 1, antiVerbatim: avDefault,
	}
}

// Option configures a Generator at construction time.
type Option func(*config)

// Order sets the context scale, 2..8. Default 4. Generation uses
// variable-length contexts up to Order+2 tokens (longest first, backing
// off where the corpus is thin), and anti-verbatim protection defaults
// to window Order+2.
//
// Mapping from text.Order to the underlying interned.BackoffConfig:
//
//	| text.Order | interned.MaxOrder | Default Steering Window |
//	|------------|-------------------|-------------------------|
//	| 2          | 4                 | 4                       |
//	| 3          | 5                 | 5                       |
//	| 4 (def)    | 6                 | 6                       |
//	| 5          | 7                 | 7                       |
//	| 6..8       | 8                 | 8                       |
func Order(n int) Option { return func(c *config) { c.order = n } }

// MinOrder sets the shortest context the engine may use, in tokens. Default 1.
// It must be between 1 and min(Order()+2, 8). Generation fails instead of
// falling below this floor when evidence or allowed continuations run out.
func MinOrder(n int) Option { return func(c *config) { c.minOrder = n } }

// MinSupport requires at least n observations of a context before using it.
// Default 1; range 1..65535. This applies to every allowed context order,
// including MinOrder. Raising it can reject otherwise observed continuations.
func MinSupport(n int) Option { return func(c *config) { c.minSupport = n } }

// Timeout is the per-Generate time budget applied to the context. Default
// 10s; 0 means no added timeout (the caller's context still applies).
func Timeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// Retries is the number of generation attempts per Generate call before it
// gives up with the last validation failure. Default 64. With Threads(t)
// each retry fans out t workers, so one Generate call can make up to
// Retries × t generation attempts.
//
// Under the steered engine most verbatim rejection happens during the
// walk itself, so far fewer retries are consumed than under a
// generate-reject-retry loop; the default stays as headroom for steering
// dead-ends and the whole-message check.
func Retries(n int) Option { return func(c *config) { c.retries = n } }

// Threads fans each attempt out across n goroutines (WithParallelism) and
// takes the first result that clears the validators. Default 0 (sequential).
// Combined with Retries(r), one Generate call can make up to r × n
// generation attempts. Values greater than 1 are rejected with RNG because
// parallel workers would share one random source.
func Threads(n int) Option { return func(c *config) { c.threads = n } }

// AntiVerbatim sets the n-gram window for verbatim-copy rejection. Default
// Order()+2. 0 disables anti-verbatim entirely (both the sliding n-gram
// check and the whole-message check).
func AntiVerbatim(n int) Option { return func(c *config) { c.antiVerbatim = n } }

// MaxTokens sets a default maximum token length for generated outputs.
// Attempts exceeding this limit are rejected and retried up to Retries.
// 0 means no maximum length limit.
func MaxTokens(n int) Option { return func(c *config) { c.maxTokens = n } }

// Bidirectional builds both a forward and a reverse engine, enabling
// middle-out seeded generation via SeedMiddle.
func Bidirectional() Option { return func(c *config) { c.bidirectional = true } }

// RNG fixes the random source for deterministic output. A Generator with a
// fixed RNG is no longer safe for concurrent Generate calls, and should not
// be combined with Threads greater than 1. Middle-out generation uses the RNG
// sequentially for reproducible results.
func RNG(r *rand.Rand) Option { return func(c *config) { c.rng = r } }

// GenOption configures a single Generate call.
type GenOption func(*genConfig)

type genConfig struct {
	generationConfig
	seed, seedMiddle []string
	verbatimFallback bool
	escape           float64
	noSteering       bool
	seedEnding       bool
	spliceCheck      bool
	noFragments      bool
}

type generationConfig struct {
	maxTokens        int
	timeout          time.Duration
	retries, threads int
}

func (c generationConfig) validate(fixedRNG bool) error {
	if c.retries < 1 {
		return fmt.Errorf("text: retries must be >= 1 (got %d)", c.retries)
	}
	if c.threads < 0 {
		return fmt.Errorf("text: threads must be >= 0 (got %d)", c.threads)
	}
	if c.maxTokens < 0 {
		return fmt.Errorf("text: maximum tokens must be >= 0 (got %d)", c.maxTokens)
	}
	if c.timeout < 0 {
		return fmt.Errorf("text: timeout must be >= 0 (got %s)", c.timeout)
	}
	if fixedRNG && c.threads > 1 {
		return errors.New("text: a fixed RNG cannot be shared by parallel workers; use Threads(0) or Threads(1)")
	}
	return nil
}

// Seed biases generation to start from the given words. Words not in the
// corpus vocabulary (tested exact, then lowercased) are ignored; if none
// remain, generation is unseeded. The surviving seed words lead the output.
func Seed(words []string) GenOption {
	return func(gc *genConfig) { gc.seed = words }
}

// SeedMiddle generates a sentence around the given words (middle-out).
// Requires the Generator to have been constructed with Bidirectional().
// Short seeds acquire observed history without lowering MinOrder or MinSupport.
// Output must add words after the seed. A backward-only candidate is discarded;
// remaining attempts generate forward-only from the same seed, within the
// original retry and timeout budgets. If no seed words survive vocabulary
// lookup, generation falls back to unseeded.
func SeedMiddle(words []string) GenOption {
	return func(gc *genConfig) { gc.seedMiddle = words }
}

// WithMaxTokens overrides the maximum token length for this generation call.
func WithMaxTokens(n int) GenOption {
	return func(gc *genConfig) { gc.maxTokens = n }
}

// WithTimeout overrides the per-Generate timeout for this call. d must be nonnegative.
func WithTimeout(d time.Duration) GenOption {
	return func(gc *genConfig) { gc.timeout = d }
}

// WithRetries overrides the retry budget for this call. n must be positive.
func WithRetries(n int) GenOption {
	return func(gc *genConfig) { gc.retries = n }
}

// WithThreads overrides the parallelism width for this call. n must be nonnegative.
func WithThreads(n int) GenOption {
	return func(gc *genConfig) { gc.threads = n }
}

// WithVerbatimFallback retries an exhausted validation failure with copy
// prevention disabled. The fallback has the same attempt limit and shares the
// original timeout; seed, context, support and length constraints still apply.
// It does not change the model or other concurrent calls.
func WithVerbatimFallback() GenOption {
	return func(gc *genConfig) { gc.verbatimFallback = true }
}

// WithEscape interpolates across context orders instead of always using the
// longest one: at each step the walk moves to the next shorter context with
// probability scale·T/(N+T), where T counts the context's distinct followers
// and N its observations (PPM method C). Recombination then happens where the
// corpus branches, including early in a message, and every transition is
// still observed. Start padding counts once, and MinOrder still applies.
// scale 0 keeps longest-context sampling.
func WithEscape(scale float64) GenOption {
	return func(gc *genConfig) { gc.escape = scale }
}

// WithoutCopySteering stops steering the walk away from verbatim n-grams
// while still rejecting outputs that repeat a whole corpus message.
func WithoutCopySteering() GenOption {
	return func(gc *genConfig) { gc.noSteering = true }
}

// WithSeedEnding lets SeedMiddle output end on the seed: when the forward
// walk stops right after the seed, a valid backward completion is kept
// instead of being discarded.
func WithSeedEnding() GenOption {
	return func(gc *genConfig) { gc.seedEnding = true }
}

// WithSpliceValidation makes SeedMiddle reject a joined output that contains
// a copied run as long as the copy-steering window (which can form across the
// seed, where neither half's steering looks) and resample both halves,
// instead of falling back to the forward half alone.
func WithSpliceValidation() GenOption {
	return func(gc *genConfig) { gc.spliceCheck = true }
}

// WithoutFragments rejects outputs that are an unbroken piece of one corpus
// message, not only whole messages: every output must recombine at least
// once. Checked exactly for outputs up to the model's longest context.
func WithoutFragments() GenOption {
	return func(gc *genConfig) { gc.noFragments = true }
}
