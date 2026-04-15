package stuck

import (
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
	c := NewCacheWithOptions(1, time.Millisecond)

	c.RecordFailure("state1")
	time.Sleep(5 * time.Millisecond)

	if c.IsStuck("state1") {
		t.Error("expected entry to have expired")
	}
}

func TestCacheUnknownState(t *testing.T) {
	c := NewCache()
	if c.IsStuck("unknown") {
		t.Error("expected unknown state to not be stuck")
	}
}
