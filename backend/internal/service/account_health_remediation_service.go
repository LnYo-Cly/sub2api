package service

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/claude"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
)

const (
	accountHealthInitialCooldown  = 10 * time.Minute
	accountHealthTransientBackoff = 30 * time.Minute
	accountHealthJobTimeout       = 3 * time.Minute
)

// AccountHealthTrigger describes why an account should be removed from
// scheduling and remediated in the background.
type AccountHealthTrigger struct {
	StatusCode int
	Reason     string
	Body       []byte
}

type accountHealthRefreshService interface {
	RefreshAccountNow(ctx context.Context, account *Account) error
}

type accountHealthTestService interface {
	RunTestBackground(ctx context.Context, accountID int64, modelID string) (*ScheduledTestResult, error)
}

type accountHealthRecoveryService interface {
	RecoverAccountAfterSuccessfulTest(ctx context.Context, accountID int64) (*SuccessfulTestRecoveryResult, error)
}

// AccountHealthRemediationService automatically handles accounts that start
// failing on the upstream path. It immediately takes the account out of
// scheduling, then refreshes OAuth credentials and runs a background probe to
// decide whether to recover or permanently disable the account.
type AccountHealthRemediationService struct {
	accountRepo         AccountRepository
	tokenRefreshService accountHealthRefreshService
	accountTestService  accountHealthTestService
	rateLimitService    accountHealthRecoveryService

	mu      sync.Mutex
	running map[int64]struct{}
}

func NewAccountHealthRemediationService(
	accountRepo AccountRepository,
	tokenRefreshService *TokenRefreshService,
	accountTestService *AccountTestService,
	rateLimitService *RateLimitService,
) *AccountHealthRemediationService {
	return &AccountHealthRemediationService{
		accountRepo:         accountRepo,
		tokenRefreshService: tokenRefreshService,
		accountTestService:  accountTestService,
		rateLimitService:    rateLimitService,
		running:             make(map[int64]struct{}),
	}
}

func (s *AccountHealthRemediationService) Trigger(ctx context.Context, account *Account, trigger AccountHealthTrigger) {
	if s == nil || s.accountRepo == nil || account == nil || account.ID <= 0 {
		return
	}
	if !s.shouldRemediate(account, trigger) {
		return
	}

	until := time.Now().Add(accountHealthInitialCooldown)
	reason := buildAccountHealthReason(trigger)
	if err := s.accountRepo.SetTempUnschedulable(ctx, account.ID, until, reason); err != nil {
		slog.Warn("account_health.temp_unsched_failed",
			"account_id", account.ID,
			"status_code", trigger.StatusCode,
			"error", err,
		)
		return
	}

	if !s.markRunning(account.ID) {
		slog.Debug("account_health.already_running", "account_id", account.ID)
		return
	}

	accountID := account.ID
	go func() {
		defer s.clearRunning(accountID)

		jobCtx, cancel := context.WithTimeout(context.Background(), accountHealthJobTimeout)
		defer cancel()

		s.run(jobCtx, accountID, trigger)
	}()
}

func (s *AccountHealthRemediationService) shouldRemediate(account *Account, trigger AccountHealthTrigger) bool {
	if account.Status == StatusError || !account.Schedulable {
		return false
	}
	if isNonRetryableAccountHealthError(trigger.StatusCode, trigger.Reason, trigger.Body) {
		return true
	}
	if trigger.StatusCode == 0 {
		return true
	}
	if trigger.StatusCode == http.StatusBadGateway || trigger.StatusCode == http.StatusServiceUnavailable || trigger.StatusCode >= 500 {
		return true
	}
	return false
}

func (s *AccountHealthRemediationService) run(ctx context.Context, accountID int64, trigger AccountHealthTrigger) {
	account, err := s.accountRepo.GetByID(ctx, accountID)
	if err != nil || account == nil {
		slog.Warn("account_health.load_failed", "account_id", accountID, "error", err)
		return
	}

	if account.Status == StatusError || !account.Schedulable {
		return
	}

	if shouldDisableBeforeRefresh(account, trigger) {
		s.disable(ctx, account, "upstream reported non-recoverable account error", trigger)
		return
	}

	if account.IsOAuth() && s.tokenRefreshService != nil {
		if err := s.tokenRefreshService.RefreshAccountNow(ctx, account); err != nil {
			if isNonRetryableRefreshError(err) {
				s.disable(ctx, account, "token refresh failed with non-retryable error", AccountHealthTrigger{
					StatusCode: trigger.StatusCode,
					Reason:     err.Error(),
					Body:       trigger.Body,
				})
				return
			}
			s.keepTempUnschedulable(ctx, account, fmt.Sprintf("token refresh transient failure: %v", err))
			return
		}

		refreshed, err := s.accountRepo.GetByID(ctx, accountID)
		if err == nil && refreshed != nil {
			account = refreshed
		}
	}

	if s.accountTestService == nil {
		s.keepTempUnschedulable(ctx, account, "account health probe unavailable")
		return
	}

	result, err := s.accountTestService.RunTestBackground(ctx, account.ID, defaultHealthProbeModel(account))
	if err != nil {
		s.keepTempUnschedulable(ctx, account, fmt.Sprintf("account health probe failed: %v", err))
		return
	}
	if result != nil && result.Status == "success" {
		s.recover(ctx, account)
		return
	}

	errMsg := ""
	if result != nil {
		errMsg = result.ErrorMessage
	}
	if isNonRetryableAccountHealthError(0, errMsg, nil) {
		s.disable(ctx, account, "account health probe reported non-recoverable error", AccountHealthTrigger{
			Reason: errMsg,
		})
		return
	}

	if errMsg == "" {
		errMsg = "account health probe failed"
	}
	s.keepTempUnschedulable(ctx, account, errMsg)
}

func (s *AccountHealthRemediationService) recover(ctx context.Context, account *Account) {
	if s.rateLimitService != nil {
		if _, err := s.rateLimitService.RecoverAccountAfterSuccessfulTest(ctx, account.ID); err != nil {
			slog.Warn("account_health.recover_failed", "account_id", account.ID, "error", err)
			return
		}
	} else if err := s.accountRepo.ClearTempUnschedulable(ctx, account.ID); err != nil {
		slog.Warn("account_health.clear_temp_unsched_failed", "account_id", account.ID, "error", err)
		return
	}
	if s.accountRepo != nil {
		if err := s.accountRepo.ClearTempUnschedulable(ctx, account.ID); err != nil {
			slog.Warn("account_health.clear_temp_unsched_failed", "account_id", account.ID, "error", err)
			return
		}
	}

	slog.Info("account_health.recovered", "account_id", account.ID)
}

func (s *AccountHealthRemediationService) keepTempUnschedulable(ctx context.Context, account *Account, reason string) {
	until := time.Now().Add(accountHealthTransientBackoff)
	msg := truncateAccountHealthMessage("auto health check inconclusive: "+strings.TrimSpace(reason), 1024)
	if err := s.accountRepo.SetTempUnschedulable(ctx, account.ID, until, msg); err != nil {
		slog.Warn("account_health.extend_temp_unsched_failed",
			"account_id", account.ID,
			"error", err,
		)
		return
	}
	slog.Warn("account_health.temp_unsched_extended",
		"account_id", account.ID,
		"until", until.Format(time.RFC3339),
		"reason", msg,
	)
}

func (s *AccountHealthRemediationService) disable(ctx context.Context, account *Account, summary string, trigger AccountHealthTrigger) {
	msg := summary
	detail := strings.TrimSpace(trigger.Reason)
	if detail == "" && len(trigger.Body) > 0 {
		detail = extractUpstreamErrorMessage(trigger.Body)
	}
	if detail != "" {
		msg += ": " + detail
	}
	if trigger.StatusCode > 0 {
		msg = fmt.Sprintf("%s (status=%d)", msg, trigger.StatusCode)
	}
	msg = truncateAccountHealthMessage(msg, 2048)

	if err := s.accountRepo.SetError(ctx, account.ID, msg); err != nil {
		slog.Warn("account_health.set_error_failed", "account_id", account.ID, "error", err)
		return
	}
	slog.Warn("account_health.disabled", "account_id", account.ID, "reason", msg)
}

func (s *AccountHealthRemediationService) markRunning(accountID int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.running[accountID]; exists {
		return false
	}
	s.running[accountID] = struct{}{}
	return true
}

func (s *AccountHealthRemediationService) clearRunning(accountID int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.running, accountID)
}

func buildAccountHealthReason(trigger AccountHealthTrigger) string {
	reason := strings.TrimSpace(trigger.Reason)
	if reason == "" && len(trigger.Body) > 0 {
		reason = extractUpstreamErrorMessage(trigger.Body)
	}
	if reason == "" {
		reason = "upstream failure"
	}
	if trigger.StatusCode > 0 {
		reason = fmt.Sprintf("auto health check pending: status=%d %s", trigger.StatusCode, reason)
	} else {
		reason = "auto health check pending: " + reason
	}
	return truncateAccountHealthMessage(reason, 1024)
}

func defaultHealthProbeModel(account *Account) string {
	if account == nil {
		return ""
	}
	switch account.Platform {
	case PlatformOpenAI:
		return openai.DefaultTestModel
	case PlatformAnthropic:
		return claude.DefaultTestModel
	default:
		return ""
	}
}

func isNonRetryableAccountHealthError(statusCode int, reason string, body []byte) bool {
	if statusCode == http.StatusUnauthorized || statusCode == http.StatusPaymentRequired {
		return true
	}
	if statusCode == http.StatusBadRequest {
		msg := strings.ToLower(reason)
		if len(body) > 0 {
			msg += " " + strings.ToLower(extractUpstreamErrorMessage(body))
			msg += " " + strings.ToLower(string(body))
		}
		return strings.Contains(msg, "organization has been disabled") ||
			strings.Contains(msg, "credit balance") ||
			strings.Contains(msg, "identity verification is required")
	}

	msg := strings.ToLower(strings.TrimSpace(reason))
	if len(body) > 0 {
		msg += " " + strings.ToLower(extractUpstreamErrorMessage(body))
		msg += " " + strings.ToLower(string(body))
	}
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return false
	}

	needles := []string{
		"invalid_grant",
		"invalid_client",
		"unauthorized_client",
		"access_denied",
		"no refresh token available",
		"token_revoked",
		"token revoked",
		"token_invalidated",
		"deactivated_workspace",
		"workspace deactivated",
		"organization has been disabled",
		"identity verification is required",
		"authentication failed (401)",
		"unauthorized (401)",
		"invalid api key",
		"incorrect api key",
		"api key revoked",
		"credit balance exhausted",
	}
	for _, needle := range needles {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

func shouldDisableBeforeRefresh(account *Account, trigger AccountHealthTrigger) bool {
	if account != nil && account.IsOAuth() && trigger.StatusCode == http.StatusUnauthorized {
		return false
	}
	return isNonRetryableAccountHealthError(trigger.StatusCode, trigger.Reason, trigger.Body)
}

func truncateAccountHealthMessage(msg string, maxLen int) string {
	msg = strings.TrimSpace(msg)
	if maxLen <= 0 || len(msg) <= maxLen {
		return msg
	}
	return msg[:maxLen]
}
