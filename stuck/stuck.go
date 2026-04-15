package stuck

import (
	"sync"
	"sync/atomic"
	"time"
)

// Cache tracks states that have been empirically determined to be stuck.
// A state is marked stuck after exceeding a failure threshold within the TTL window.
// Implements barkov.StuckDetector.
type Cache struct {
	cache      sync.Map
	maxFails   int
	expireTime time.Duration
}

type entry struct {
	fails     atomic.Int32
	timestamp time.Time
	stuck     atomic.Bool
}

// NewCache creates a stuck-state cache with sensible defaults
// (20 failures to mark stuck, 1 hour TTL).
func NewCache() *Cache {
	return &Cache{
		maxFails:   20,
		expireTime: time.Hour,
	}
}

// NewCacheWithOptions creates a stuck-state cache with custom settings.
func NewCacheWithOptions(maxFails int, ttl time.Duration) *Cache {
	return &Cache{
		maxFails:   maxFails,
		expireTime: ttl,
	}
}

// RecordFailure records a generation failure for the given encoded state.
// Returns true if the state should now be considered stuck.
func (c *Cache) RecordFailure(state string) bool {
	e := c.getOrCreate(state)
	if time.Since(e.timestamp) > c.expireTime {
		e.fails.Store(0)
		e.stuck.Store(false)
		e.timestamp = time.Now()
	}
	if int(e.fails.Add(1)) >= c.maxFails {
		e.stuck.Store(true)
		return true
	}
	return false
}

// RecordSuccess resets the failure count for a state.
func (c *Cache) RecordSuccess(state string) {
	c.cache.Delete(state)
}

// IsStuck returns true if the state has been marked as stuck and hasn't expired.
func (c *Cache) IsStuck(state string) bool {
	v, ok := c.cache.Load(state)
	if !ok {
		return false
	}
	e := v.(*entry)
	if time.Since(e.timestamp) > c.expireTime {
		return false
	}
	return e.stuck.Load()
}

func (c *Cache) getOrCreate(state string) *entry {
	v, _ := c.cache.LoadOrStore(state, &entry{timestamp: time.Now()})
	return v.(*entry)
}
