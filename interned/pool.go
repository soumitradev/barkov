package interned

import "sync"

// TokenIDPool implements barkov.SlicePool[TokenID] using sync.Pool.
type TokenIDPool struct {
	statePool     sync.Pool
	generatedPool sync.Pool
}

// NewTokenIDPool creates a pool for reusing TokenID slices.
func NewTokenIDPool() *TokenIDPool {
	return &TokenIDPool{
		statePool: sync.Pool{
			New: func() any { s := make([]TokenID, 0, 16); return &s },
		},
		generatedPool: sync.Pool{
			New: func() any { s := make([]TokenID, 0, 128); return &s },
		},
	}
}

func (p *TokenIDPool) GetState() *[]TokenID {
	return p.statePool.Get().(*[]TokenID)
}

func (p *TokenIDPool) PutState(s *[]TokenID) {
	*s = (*s)[:0]
	p.statePool.Put(s)
}

func (p *TokenIDPool) GetGenerated() *[]TokenID {
	return p.generatedPool.Get().(*[]TokenID)
}

func (p *TokenIDPool) PutGenerated(s *[]TokenID) {
	*s = (*s)[:0]
	p.generatedPool.Put(s)
}
