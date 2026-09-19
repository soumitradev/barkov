// Package stuck tracks seeds that repeatedly fail within a time window.
package stuck

import (
	"sync"
	"time"
)

// Cache counts failures within each state's TTL. Methods are safe for concurrent
// use; success clears the state, and expired entries are removed on access.
type Cache struct {
	mu         sync.Mutex
	entries    map[string]entry
	maxFails   int
	expireTime time.Duration
}

type entry struct {
	failures int
	started  time.Time
}

// NewCache uses a threshold of 20 failures and a one-hour TTL.
func NewCache() *Cache { return NewCacheWithOptions(20, time.Hour) }

// NewCacheWithOptions creates a cache with the given failure threshold and TTL.
func NewCacheWithOptions(maxFails int, ttl time.Duration) *Cache {
	return &Cache{entries: make(map[string]entry), maxFails: maxFails, expireTime: ttl}
}

// RecordFailure records one failure and reports whether the threshold was reached.
func (c *Cache) RecordFailure(state string) bool { return c.recordFailureAt(state, time.Now()) }

func (c *Cache) recordFailureAt(state string, now time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[state]
	if !ok || now.Sub(e.started) > c.expireTime {
		e = entry{started: now}
	}
	e.failures++
	if c.entries == nil {
		c.entries = make(map[string]entry)
	}
	c.entries[state] = e
	return e.failures >= c.maxFails
}

// RecordSuccess clears previous failures for the state.
func (c *Cache) RecordSuccess(state string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.entries, state)
}

// IsStuck reports whether the state has reached its threshold within the TTL.
func (c *Cache) IsStuck(state string) bool { return c.isStuckAt(state, time.Now()) }

func (c *Cache) isStuckAt(state string, now time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[state]
	if !ok {
		return false
	}
	if now.Sub(e.started) > c.expireTime {
		delete(c.entries, state)
		return false
	}
	return e.failures >= c.maxFails
}
