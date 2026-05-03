//go:build unit

package service

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type healthRemediationRepo struct {
	mockAccountRepoForGemini

	mu                 sync.Mutex
	setTempCalls       int
	clearTempCalls     int
	setErrorCalls      int
	lastTempReason     string
	lastErrorMessage   string
	tempSetCh          chan struct{}
	clearTempCh        chan struct{}
	setErrorCh         chan struct{}
	sleepOnSetTempOnce time.Duration
}

func newHealthRemediationRepo(account *Account) *healthRemediationRepo {
	return &healthRemediationRepo{
		mockAccountRepoForGemini: mockAccountRepoForGemini{
			accountsByID: map[int64]*Account{account.ID: account},
		},
		tempSetCh:   make(chan struct{}, 16),
		clearTempCh: make(chan struct{}, 16),
		setErrorCh:  make(chan struct{}, 16),
	}
}

func (r *healthRemediationRepo) SetTempUnschedulable(ctx context.Context, id int64, until time.Time, reason string) error {
	r.mu.Lock()
	r.setTempCalls++
	call := r.setTempCalls
	r.lastTempReason = reason
	if acc := r.accountsByID[id]; acc != nil {
		acc.TempUnschedulableUntil = &until
		acc.TempUnschedulableReason = reason
	}
	sleep := time.Duration(0)
	if call == 1 {
		sleep = r.sleepOnSetTempOnce
	}
	r.mu.Unlock()

	if sleep > 0 {
		time.Sleep(sleep)
	}
	select {
	case r.tempSetCh <- struct{}{}:
	default:
	}
	return nil
}

func (r *healthRemediationRepo) ClearTempUnschedulable(ctx context.Context, id int64) error {
	r.mu.Lock()
	r.clearTempCalls++
	if acc := r.accountsByID[id]; acc != nil {
		acc.TempUnschedulableUntil = nil
		acc.TempUnschedulableReason = ""
	}
	r.mu.Unlock()
	select {
	case r.clearTempCh <- struct{}{}:
	default:
	}
	return nil
}

func (r *healthRemediationRepo) SetError(ctx context.Context, id int64, errorMsg string) error {
	r.mu.Lock()
	r.setErrorCalls++
	r.lastErrorMessage = errorMsg
	if acc := r.accountsByID[id]; acc != nil {
		acc.Status = StatusError
		acc.ErrorMessage = errorMsg
	}
	r.mu.Unlock()
	select {
	case r.setErrorCh <- struct{}{}:
	default:
	}
	return nil
}

func (r *healthRemediationRepo) snapshot() (setTemp, clearTemp, setError int, tempReason, errorMessage string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.setTempCalls, r.clearTempCalls, r.setErrorCalls, r.lastTempReason, r.lastErrorMessage
}

type healthRefreshStub struct {
	mu    sync.Mutex
	calls int
	err   error
}

func (s *healthRefreshStub) RefreshAccountNow(ctx context.Context, account *Account) error {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return s.err
}

func (s *healthRefreshStub) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

type healthTestStub struct {
	mu     sync.Mutex
	calls  int
	result *ScheduledTestResult
	err    error
}

func (s *healthTestStub) RunTestBackground(ctx context.Context, accountID int64, modelID string) (*ScheduledTestResult, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return s.result, s.err
}

func (s *healthTestStub) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

type healthRecoveryStub struct {
	mu    sync.Mutex
	calls int
	err   error
}

func (s *healthRecoveryStub) RecoverAccountAfterSuccessfulTest(ctx context.Context, accountID int64) (*SuccessfulTestRecoveryResult, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return &SuccessfulTestRecoveryResult{ClearedRateLimit: true}, s.err
}

func (s *healthRecoveryStub) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func TestAccountHealthRemediation_RefreshSuccessProbeSuccessRecovers(t *testing.T) {
	account := &Account{ID: 42, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true}
	repo := newHealthRemediationRepo(account)
	refresh := &healthRefreshStub{}
	probe := &healthTestStub{result: &ScheduledTestResult{Status: "success"}}
	recovery := &healthRecoveryStub{}

	svc := &AccountHealthRemediationService{
		accountRepo:         repo,
		tokenRefreshService: refresh,
		accountTestService:  probe,
		rateLimitService:    recovery,
		running:             make(map[int64]struct{}),
	}

	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 502, Reason: "Upstream request failed"})

	require.Eventually(t, func() bool {
		_, clearTemp, _, _, _ := repo.snapshot()
		return clearTemp == 1 && recovery.callCount() == 1
	}, time.Second, 10*time.Millisecond)
	require.Equal(t, 1, refresh.callCount())
	require.Equal(t, 1, probe.callCount())
	_, clearTemp, setError, _, _ := repo.snapshot()
	require.Equal(t, 1, clearTemp)
	require.Equal(t, 0, setError)
}

func TestAccountHealthRemediation_NonRetryableRefreshDisables(t *testing.T) {
	account := &Account{ID: 43, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true}
	repo := newHealthRemediationRepo(account)
	refresh := &healthRefreshStub{err: errors.New("invalid_grant: token revoked")}
	probe := &healthTestStub{result: &ScheduledTestResult{Status: "success"}}
	recovery := &healthRecoveryStub{}

	svc := &AccountHealthRemediationService{
		accountRepo:         repo,
		tokenRefreshService: refresh,
		accountTestService:  probe,
		rateLimitService:    recovery,
		running:             make(map[int64]struct{}),
	}

	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 502, Reason: "Upstream request failed"})

	require.Eventually(t, func() bool {
		_, _, setError, _, _ := repo.snapshot()
		return setError == 1
	}, time.Second, 10*time.Millisecond)
	require.Equal(t, 1, refresh.callCount())
	require.Equal(t, 0, probe.callCount())
	_, _, _, _, errorMsg := repo.snapshot()
	require.Contains(t, errorMsg, "invalid_grant")
}

func TestAccountHealthRemediation_OAuth401RefreshSuccessCanRecover(t *testing.T) {
	account := &Account{ID: 46, Platform: PlatformOpenAI, Type: AccountTypeOAuth, Status: StatusActive, Schedulable: true}
	repo := newHealthRemediationRepo(account)
	refresh := &healthRefreshStub{}
	probe := &healthTestStub{result: &ScheduledTestResult{Status: "success"}}
	recovery := &healthRecoveryStub{}

	svc := &AccountHealthRemediationService{
		accountRepo:         repo,
		tokenRefreshService: refresh,
		accountTestService:  probe,
		rateLimitService:    recovery,
		running:             make(map[int64]struct{}),
	}

	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 401, Reason: "expired access token"})

	require.Eventually(t, func() bool {
		_, clearTemp, _, _, _ := repo.snapshot()
		return clearTemp == 1 && recovery.callCount() == 1
	}, time.Second, 10*time.Millisecond)
	require.Equal(t, 1, refresh.callCount())
	require.Equal(t, 1, probe.callCount())
	_, _, setError, _, _ := repo.snapshot()
	require.Equal(t, 0, setError)
}

func TestAccountHealthRemediation_TransientProbeFailureKeepsTempUnschedulable(t *testing.T) {
	account := &Account{ID: 44, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Status: StatusActive, Schedulable: true}
	repo := newHealthRemediationRepo(account)
	probe := &healthTestStub{result: &ScheduledTestResult{Status: "failed", ErrorMessage: "upstream timeout"}}

	svc := &AccountHealthRemediationService{
		accountRepo:        repo,
		accountTestService: probe,
		running:            make(map[int64]struct{}),
	}

	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 503, Reason: "temporary unavailable"})

	require.Eventually(t, func() bool {
		setTemp, _, setError, _, _ := repo.snapshot()
		return setTemp == 2 && setError == 0
	}, time.Second, 10*time.Millisecond)
	_, clearTemp, setError, reason, _ := repo.snapshot()
	require.Equal(t, 0, clearTemp)
	require.Equal(t, 0, setError)
	require.Contains(t, reason, "inconclusive")
}

func TestAccountHealthRemediation_DuplicateTriggersCoalesce(t *testing.T) {
	account := &Account{ID: 45, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Status: StatusActive, Schedulable: true}
	repo := newHealthRemediationRepo(account)
	repo.sleepOnSetTempOnce = 50 * time.Millisecond
	probe := &healthTestStub{result: &ScheduledTestResult{Status: "success"}}

	svc := &AccountHealthRemediationService{
		accountRepo:        repo,
		accountTestService: probe,
		running:            make(map[int64]struct{}),
	}

	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 502, Reason: "first"})
	<-repo.tempSetCh
	svc.Trigger(context.Background(), account, AccountHealthTrigger{StatusCode: 502, Reason: "second"})

	require.Eventually(t, func() bool {
		return probe.callCount() == 1
	}, time.Second, 10*time.Millisecond)
	require.Never(t, func() bool {
		return probe.callCount() > 1
	}, 120*time.Millisecond, 10*time.Millisecond)
}
