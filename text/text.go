// Package text is the sugared, string-first layer over barkov: corpus of
// tokenized messages in, generated sentences out, with the fastest engine
// and anti-verbatim protection wired up for you.
//
// It is to the core packages what a facade is to a toolkit. The core gives
// you generics, pluggable encoders, and a dozen functional options; text
// pins the token type to string, picks interned.BuildBackoff as the
// engine (variable-order contexts with verbatim steering built into the
// walk), and absorbs the retry-and-timeout ceremony that every real
// consumer ends up writing by hand. When you need control, drop down:
// Chain() and Vocab() hand you the underlying core objects.
//
// text imports only the standard library and barkov's own zero-dependency
// packages, so adopting it never pulls an external module into your build.
package text

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"

	barkov "github.com/soumitradev/barkov/v2"
	"github.com/soumitradev/barkov/v2/interned"
)

// avDefault is the sentinel config value meaning "anti-verbatim window not
// set; default it to Order()+2". A user-supplied 0 disables anti-verbatim;
// a positive value fixes the window.
const avDefault = -1

type config struct {
	order         int
	timeout       time.Duration
	retries       int
	threads       int
	antiVerbatim  int
	maxTokens     int
	bidirectional bool
	rng           *rand.Rand
}

func defaultConfig() config {
	return config{
		order:         4,
		timeout:       10 * time.Second,
		retries:       64,
		threads:       0,
		antiVerbatim:  avDefault,
		maxTokens:     0,
		bidirectional: false,
		rng:           nil,
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
// generation attempts. Do not combine with RNG: parallel workers share
// one random source.
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
// be combined with Threads.
func RNG(r *rand.Rand) Option { return func(c *config) { c.rng = r } }

// Generator is a ready-to-use, string-first text generator. Unless RNG was
// set, its Generate/Sentence methods are safe to call concurrently.
type Generator struct {
	chain        barkov.GenerativeChain[interned.TokenID]
	reverseChain barkov.GenerativeChain[interned.TokenID]
	vocab        *interned.Vocabulary
	timeout      time.Duration
	retries      int
	threads      int
	maxTokens    int

	// messages holds fnv digests of every complete corpus message, so a
	// short output that reproduces a whole message is caught even when it
	// is too short for any n-gram window. nil when anti-verbatim is disabled.
	messages *interned.MessageSet
}

// New builds a Generator from a pre-tokenized corpus (each inner slice is
// one message's tokens — text does no tokenization itself). It builds the
// vocabulary, interns the corpus, and constructs the engine internally:
// a variable-order backoff chain that samples where the corpus genuinely
// branches and steers around verbatim reproductions during the walk. It
// returns an error on an empty or unusable corpus or an invalid option.
func New(corpus [][]string, opts ...Option) (*Generator, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.order < 2 || cfg.order > 8 {
		return nil, fmt.Errorf("text: Order(%d) out of range 2..8", cfg.order)
	}
	if cfg.retries < 1 {
		return nil, fmt.Errorf("text: Retries(%d) must be >= 1", cfg.retries)
	}
	if cfg.threads < 0 {
		return nil, fmt.Errorf("text: Threads(%d) must be >= 0", cfg.threads)
	}
	if cfg.antiVerbatim < 0 && cfg.antiVerbatim != avDefault {
		return nil, fmt.Errorf("text: AntiVerbatim(%d) must be >= 0", cfg.antiVerbatim)
	}
	if cfg.maxTokens < 0 {
		return nil, fmt.Errorf("text: MaxTokens(%d) must be >= 0", cfg.maxTokens)
	}
	if !hasContent(corpus) {
		return nil, errors.New("text: corpus has no usable messages")
	}

	maxOrder := min(cfg.order+2, 8)
	maxVerbatim := -1 // AntiVerbatim(0): steering off
	if cfg.antiVerbatim != 0 {
		window := cfg.antiVerbatim
		if window == avDefault {
			window = cfg.order + 2
		}
		maxVerbatim = max(2, min(window, maxOrder))
	}

	vocab := interned.NewVocabulary()
	encoded := vocab.InternCorpus(corpus)

	var (
		chain    barkov.GenerativeChain[interned.TokenID]
		revChain barkov.GenerativeChain[interned.TokenID]
		messages *interned.MessageSet
		wg       sync.WaitGroup
	)

	wg.Add(1)
	go func() {
		defer wg.Done()
		chain = interned.BuildBackoff(interned.BackoffConfig{
			MaxOrder:    maxOrder,
			MaxVerbatim: maxVerbatim,
		}, encoded)
	}()

	if cfg.antiVerbatim != 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			messages = interned.NewMessageSet(encoded)
		}()
	}

	if cfg.bidirectional {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reversedEncoded := make([][]interned.TokenID, len(encoded))
			for i, msg := range encoded {
				rev := make([]interned.TokenID, len(msg))
				copy(rev, msg)
				slices.Reverse(rev)
				reversedEncoded[i] = rev
			}
			revChain = interned.BuildBackoff(interned.BackoffConfig{
				MaxOrder:    maxOrder,
				MaxVerbatim: maxVerbatim,
			}, reversedEncoded)
		}()
	}

	wg.Wait()

	if cfg.rng != nil {
		if s, ok := chain.(barkov.RNGSettable); ok {
			s.SetRNG(cfg.rng)
		}
		if revChain != nil {
			if s, ok := revChain.(barkov.RNGSettable); ok {
				s.SetRNG(cfg.rng)
			}
		}
	}

	g := &Generator{
		chain:        chain,
		reverseChain: revChain,
		vocab:        vocab,
		timeout:      cfg.timeout,
		retries:      cfg.retries,
		threads:      cfg.threads,
		maxTokens:    cfg.maxTokens,
		messages:     messages,
	}

	return g, nil
}

func hasContent(corpus [][]string) bool {
	for _, msg := range corpus {
		if len(msg) > 0 {
			return true
		}
	}
	return false
}

// GenOption configures a single Generate call.
type GenOption func(*genConfig)

type genConfig struct {
	seed       []string
	seedMiddle []string
	maxTokens  int
	timeout    time.Duration
	retries    int
	threads    int
}

// Seed biases generation to start from the given words. Words not in the
// corpus vocabulary (tested exact, then lowercased) are ignored; if none
// remain, generation is unseeded. The surviving seed words lead the output.
func Seed(words []string) GenOption {
	return func(gc *genConfig) { gc.seed = words }
}

// SeedMiddle generates a sentence around the given words (middle-out).
// Requires the Generator to have been constructed with Bidirectional().
// If no seed words survive vocabulary lookup, generation falls back to unseeded.
func SeedMiddle(words []string) GenOption {
	return func(gc *genConfig) { gc.seedMiddle = words }
}

// WithMaxTokens overrides the maximum token length for this generation call.
func WithMaxTokens(n int) GenOption {
	return func(gc *genConfig) { gc.maxTokens = n }
}

// WithTimeout overrides the per-Generate timeout for this call.
func WithTimeout(d time.Duration) GenOption {
	return func(gc *genConfig) { gc.timeout = d }
}

// WithRetries overrides the retry budget for this call.
func WithRetries(n int) GenOption {
	return func(gc *genConfig) { gc.retries = n }
}

// WithThreads overrides the parallelism width for this call.
func WithThreads(n int) GenOption {
	return func(gc *genConfig) { gc.threads = n }
}

// Generate returns one generated message as tokens. Seed words that survive
// vocabulary lookup lead the output (or sit mid-sentence under SeedMiddle).
// It retries on anti-verbatim or length rejection up to the configured Retries,
// and returns context and barkov errors unchanged.
func (g *Generator) Generate(ctx context.Context, opts ...GenOption) ([]string, error) {
	gc := genConfig{
		maxTokens: g.maxTokens,
		timeout:   g.timeout,
		retries:   g.retries,
		threads:   g.threads,
	}
	for _, opt := range opts {
		opt(&gc)
	}

	if gc.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, gc.timeout)
		defer cancel()
	}

	if gc.seedMiddle != nil {
		return g.generateMiddleOut(ctx, &gc)
	}

	seed := g.internSeed(gc.seed)
	return g.generateForward(ctx, &gc, seed)
}

func (g *Generator) generateForward(ctx context.Context, gc *genConfig, seed []interned.TokenID) ([]string, error) {
	genOpts := g.genOptions(seed, gc.maxTokens, gc.threads)

	var lastErr error
	for attempt := 0; attempt < gc.retries; attempt++ {
		out, err := barkov.Gen(ctx, g.chain, genOpts...)
		if err == nil {
			return g.vocab.DecodeTokens(out), nil
		}
		if !errors.Is(err, barkov.ErrSentenceFailedValidation) {
			return nil, err
		}
		lastErr = err
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	return nil, lastErr
}

func (g *Generator) generateMiddleOut(ctx context.Context, gc *genConfig) ([]string, error) {
	if g.reverseChain == nil {
		return nil, errors.New("text: SeedMiddle requires Bidirectional() option at New")
	}
	seed := g.internSeed(gc.seedMiddle)
	if len(seed) == 0 {
		return g.generateForward(ctx, gc, nil)
	}

	revSeed := make([]interned.TokenID, len(seed))
	copy(revSeed, seed)
	slices.Reverse(revSeed)

	fwdOpts := g.genOptions(seed, 0, gc.threads)
	bwdOpts := g.genOptions(revSeed, 0, gc.threads)

	var lastErr error
	for attempt := 0; attempt < gc.retries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		var (
			fwd, bwd       []interned.TokenID
			fwdErr, bwdErr error
			wg             sync.WaitGroup
		)
		wg.Add(2)
		go func() {
			defer wg.Done()
			fwd, fwdErr = barkov.Gen(ctx, g.chain, fwdOpts...)
		}()
		go func() {
			defer wg.Done()
			bwd, bwdErr = barkov.Gen(ctx, g.reverseChain, bwdOpts...)
		}()
		wg.Wait()

		if fwdErr != nil && !errors.Is(fwdErr, barkov.ErrSentenceFailedValidation) {
			return nil, fwdErr
		}
		if bwdErr != nil && !errors.Is(bwdErr, barkov.ErrSentenceFailedValidation) {
			return nil, bwdErr
		}

		var spliced []interned.TokenID
		if fwdErr == nil && bwdErr == nil {
			slices.Reverse(bwd)
			prefixLen := len(bwd) - len(seed)
			if prefixLen < 0 {
				prefixLen = 0
			}
			spliced = make([]interned.TokenID, 0, prefixLen+len(fwd))
			spliced = append(spliced, bwd[:prefixLen]...)
			spliced = append(spliced, fwd...)
		} else if fwdErr == nil {
			// Graceful fallback to forward half if backward failed
			spliced = fwd
		} else if bwdErr == nil {
			// Graceful fallback to backward half if forward failed
			slices.Reverse(bwd)
			spliced = bwd
		} else {
			lastErr = fwdErr
			continue
		}

		if gc.maxTokens > 0 && len(spliced) > gc.maxTokens {
			lastErr = fmt.Errorf("barkov: output exceeded maximum tokens (%d): %w", gc.maxTokens, barkov.ErrSentenceFailedValidation)
			continue
		}
		if g.messages != nil && g.messages.Contains(spliced) {
			lastErr = fmt.Errorf("barkov: output failed validation: %w", barkov.ErrSentenceFailedValidation)
			continue
		}

		return g.vocab.DecodeTokens(spliced), nil
	}
	return nil, lastErr
}

// Sentence is Generate joined with single spaces.
func (g *Generator) Sentence(ctx context.Context, opts ...GenOption) (string, error) {
	out, err := g.Generate(ctx, opts...)
	if err != nil {
		return "", err
	}
	return strings.Join(out, " "), nil
}

// internSeed maps seed words to TokenIDs, dropping any not in the vocabulary
// after falling back to lowercase. Returns nil (unseeded) when nothing survives.
func (g *Generator) internSeed(words []string) []interned.TokenID {
	if len(words) == 0 {
		return nil
	}
	seed := make([]interned.TokenID, 0, len(words))
	for _, w := range words {
		if id, ok := g.vocab.Lookup(w); ok {
			seed = append(seed, id)
			continue
		}
		if lw := strings.ToLower(w); lw != w {
			if id, ok := g.vocab.Lookup(lw); ok {
				seed = append(seed, id)
			}
		}
	}
	if len(seed) == 0 {
		return nil
	}
	return seed
}

// genOptions assembles the core generation options for one attempt.
func (g *Generator) genOptions(seed []interned.TokenID, maxTokens, threads int) []barkov.GenOption[interned.TokenID] {
	opts := make([]barkov.GenOption[interned.TokenID], 0, 4)
	if len(seed) > 0 {
		opts = append(opts, barkov.WithSeed(seed))
	}
	if maxTokens > 0 {
		opts = append(opts, barkov.WithMaxTokens[interned.TokenID](maxTokens))
	}
	if g.messages != nil {
		opts = append(opts, barkov.WithOutputValidator(g.messages.NotWholeMessage()))
	}
	if threads > 0 {
		opts = append(opts, barkov.WithParallelism[interned.TokenID](threads))
	}
	return opts
}

// Chain returns the underlying engine for callers who need the full core
// API (raw barkov.Gen, custom validators, state inspection). The tokens it
// speaks are interned.TokenID; use Vocab to translate.
func (g *Generator) Chain() barkov.GenerativeChain[interned.TokenID] { return g.chain }

// ReverseChain returns the reverse engine when constructed with Bidirectional(),
// or nil otherwise.
func (g *Generator) ReverseChain() barkov.GenerativeChain[interned.TokenID] { return g.reverseChain }

// Vocab returns the vocabulary that maps between strings and the TokenIDs
// the Chain uses.
func (g *Generator) Vocab() *interned.Vocabulary { return g.vocab }

// Contains reports whether word exists in the vocabulary (tested exact, then lowercased).
func (g *Generator) Contains(word string) bool {
	if g == nil || g.vocab == nil {
		return false
	}
	if _, ok := g.vocab.Lookup(word); ok {
		return true
	}
	if lw := strings.ToLower(word); lw != word {
		if _, ok := g.vocab.Lookup(lw); ok {
			return true
		}
	}
	return false
}

// SetRNG sets the random source for deterministic generation on the generator
// and any reverse engine. Not safe for concurrent use.
func (g *Generator) SetRNG(r *rand.Rand) {
	if s, ok := g.chain.(barkov.RNGSettable); ok {
		s.SetRNG(r)
	}
	if g.reverseChain != nil {
		if s, ok := g.reverseChain.(barkov.RNGSettable); ok {
			s.SetRNG(r)
		}
	}
}

// MessageSet returns the whole-message digest set built from the corpus,
// or nil if AntiVerbatim(0) was set.
func (g *Generator) MessageSet() *interned.MessageSet { return g.messages }

// States iterates all states at the given order (1..MaxOrder) as decoded
// string token slices with their observation counts.
func (g *Generator) States(order int) iter.Seq2[[]string, uint32] {
	bc, ok := g.chain.(interface {
		States(int) iter.Seq2[[]interned.TokenID, uint32]
	})
	if !ok {
		return func(yield func([]string, uint32) bool) {}
	}
	return func(yield func([]string, uint32) bool) {
		for tokens, count := range bc.States(order) {
			words := g.vocab.DecodeTokens(tokens)
			if !yield(words, count) {
				return
			}
		}
	}
}
