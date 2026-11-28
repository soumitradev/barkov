package barkov

// Config for generator
type GenOption[T comparable] func(*genConfig[T])

type genConfig[T comparable] struct {
	seed        []T
	validator   func([]T) bool
	parallelism int // 0 = single-threaded; >0 = goroutine count
}

// WithSeed generates with a seed, and pads seed with Sentinels.Begin if seed is shorter than state size
func WithSeed[T comparable](seed []T) GenOption[T] {
	return func(c *genConfig[T]) { c.seed = seed }
}

// WithValidator aborts generation when v rejects the history so far
func WithValidator[T comparable](v func([]T) bool) GenOption[T] {
	return func(c *genConfig[T]) { c.validator = v }
}

// WithParallelism sets the goroutine fan-out size, 0 (default) and 1 are single threaded
func WithParallelism[T comparable](n int) GenOption[T] {
	return func(c *genConfig[T]) { c.parallelism = n }
}
