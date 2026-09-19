package stuck

import (
	"sync"
	"testing"
	"time"
)

func TestCacheRecordFailure(t *testing.T) {
	c := NewCacheWithOptions(3, time.Minute)

	for i := range 2 {
		if c.RecordFailure("state1") {
			t.Errorf("iteration %d: expected not stuck yet", i)
		}
	}
	if !c.RecordFailure("state1") {
		t.Error("expected stuck after 3 failures")
	}
	if !c.IsStuck("state1") {
		t.Error("expected IsStuck to return true")
	}
}

func TestCacheRecordSuccess(t *testing.T) {
	c := NewCacheWithOptions(3, time.Minute)

	c.RecordFailure("state1")
	c.RecordFailure("state1")
	c.RecordFailure("state1")

	c.RecordSuccess("state1")
	if c.IsStuck("state1") {
		t.Error("expected not stuck after success")
	}
}

func TestCacheExpiry(t *testing.T) {
	c := NewCacheWithOptions(2, time.Minute)
	now := time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC)
	c.recordFailureAt("seed", now)
	if !c.recordFailureAt("seed", now) {
		t.Fatal("threshold not reached")
	}
	if !c.isStuckAt("seed", now.Add(time.Minute)) {
		t.Fatal("expired before TTL elapsed")
	}
	later := now.Add(time.Minute + time.Nanosecond)
	if c.isStuckAt("seed", later) {
		t.Fatal("stuck entry survived expiry")
	}
	if c.recordFailureAt("seed", later) {
		t.Fatal("expired failures counted toward new window")
	}
	if !c.recordFailureAt("seed", later) {
		t.Fatal("new window did not count failures")
	}
}

func TestCacheConcurrentFailuresAndExpiry(t *testing.T) {
	c := NewCacheWithOptions(800, time.Hour)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				c.RecordFailure("seed")
				c.IsStuck("seed")
			}
		}()
	}
	wg.Wait()
	if !c.IsStuck("seed") {
		t.Fatal("concurrent failures were lost")
	}
	c.RecordSuccess("seed")
	if c.IsStuck("seed") {
		t.Fatal("success did not reset failures")
	}
	c = NewCacheWithOptions(1, time.Nanosecond)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				c.RecordFailure("seed")
				c.IsStuck("seed")
			}
		}()
	}
	wg.Wait()
}

func TestCacheUnknownState(t *testing.T) {
	c := NewCache()
	if c.IsStuck("unknown") {
		t.Error("expected unknown state to not be stuck")
	}
}
