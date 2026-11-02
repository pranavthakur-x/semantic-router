package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Regression test for #5548: Claude Code appends mid-conversation role:"system"
// notices (such as <total_tokens> or file-change reminders) after tool_result
// turns. Verify the full Anthropic Messages decode -> signal extraction ->
// Router Learning protection -> Router Replay session_policy contract across
// both conversation and session protection scopes.
func TestClaudeCodeAnthropicSystemNoticeAfterToolResultPreservesToolLoopProtection(t *testing.T) {
	const (
		turn1InitialPromptWithSystemNotice = `{
			"model": "team/coding-agents",
			"max_tokens": 1024,
			"messages": [
				{"role": "user", "content": "Run uname -m"},
				{"role": "system", "content": [{"type": "text", "text": "<system-reminder>\n<total_tokens>14999000 tokens left</total_tokens>\n</system-reminder>"}]}
			]
		}`
		turn2ToolResultWithTrailingSystemNotice = `{
			"model": "team/coding-agents",
			"max_tokens": 1024,
			"messages": [
				{"role": "user", "content": "Run uname -m"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "uname -m"}}]},
				{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "aarch64", "is_error": false}]},
				{"role": "system", "content": [{"type": "text", "text": "<system-reminder>\n<total_tokens>14984549 tokens left</total_tokens>\n</system-reminder>"}]}
			]
		}`
		turn2MixedToolResultAndTextWithTrailingSystemNotice = `{
			"model": "team/coding-agents",
			"max_tokens": 1024,
			"messages": [
				{"role": "user", "content": "Run uname -m"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "uname -m"}}]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "call_1", "content": "aarch64", "is_error": false},
					{"type": "text", "text": "<system-reminder>config.yaml modified</system-reminder>"}
				]},
				{"role": "system", "content": [{"type": "text", "text": "<system-reminder>\n<total_tokens>14984549 tokens left</total_tokens>\n</system-reminder>"}]}
			]
		}`
		turn3NextUserPromptAfterCompletedToolLoop = `{
			"model": "team/coding-agents",
			"max_tokens": 1024,
			"messages": [
				{"role": "user", "content": "Run uname -m"},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "uname -m"}}]},
				{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "aarch64", "is_error": false}]},
				{"role": "system", "content": [{"type": "text", "text": "<system-reminder>\n<total_tokens>14984549 tokens left</total_tokens>\n</system-reminder>"}]},
				{"role": "assistant", "content": [{"type": "text", "text": "The host architecture is aarch64."}]},
				{"role": "user", "content": "Now explain why this matters for SIMD builds"},
				{"role": "system", "content": [{"type": "text", "text": "<system-reminder>\n<total_tokens>14980000 tokens left</total_tokens>\n</system-reminder>"}]}
			]
		}`
	)

	for _, scope := range []string{
		config.RouterLearningScopeConversation,
		config.RouterLearningScopeSession,
	} {
		for _, toolTurnVariant := range []struct {
			name string
			body string
		}{
			{name: "tool_result_then_system", body: turn2ToolResultWithTrailingSystemNotice},
			{name: "mixed_tool_result_and_user_text_then_system", body: turn2MixedToolResultAndTextWithTrailingSystemNotice},
		} {
			t.Run(scope+"/"+toolTurnVariant.name, func(t *testing.T) {
				sessiontelemetry.ResetRouterSessionMemoryForTesting()
				sessiontelemetry.ResetLastModelForTesting()
				t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
				t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

				registry := selection.NewRegistry()
				cfg := routerLearningProtectionOnlyTestConfig(scope)
				router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
				candidates := []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}}

				runTurn := func(
					turnIndex int,
					rawBody string,
					turnCandidates []config.ModelRef,
					proposedModel string,
				) (string, *RequestContext, routerreplay.RoutingRecord, error) {
					t.Helper()
					req := decodeSignalRequest(t, llmprotocol.AnthropicMessagesV1, rawBody)
					history := extractSignalConversationHistory(req)
					input := router.prepareSignalEvaluationInput(history)

					ctx := routerLearningRequestContext("claude-session-1", "claude-conv-1")
					ctx.TurnIndex = turnIndex
					ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
					ctx.SemanticRequest = req
					ctx.VSRSelectedDecision = &config.Decision{Name: "coding"}
					ctx.VSRConversationFacts = input.conversationFacts

					registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
						SelectedModel: proposedModel,
						Score:         1.0,
						Confidence:    1.0,
						Method:        selection.MethodStatic,
						AllScores:     map[string]float64{proposedModel: 1.0},
					}})

					selCtx := router.buildSelectionContext(turnCandidates, "coding", input.evaluationText, nil, "", nil, ctx)
					selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
					if err != nil {
						return "", ctx, routerreplay.RoutingRecord{}, err
					}
					if selected == nil {
						t.Fatalf("turn %d returned nil selected model without error", turnIndex)
					}
					if err := commitAgenticSessionDecision(ctx); err != nil {
						t.Fatalf("turn %d commit failed: %v", turnIndex, err)
					}
					record := buildReplayRoutingRecord(ctx, "team/coding-agents", selected.Model, "coding")
					return selected.Model, ctx, record, nil
				}

				// Turn 0: initial user prompt + system reminder establishes model-a as user_turn.
				model0, _, record0, err := runTurn(0, turn1InitialPromptWithSystemNotice, candidates, "model-a")
				if err != nil || model0 != "model-a" {
					t.Fatalf("turn 0 got (%q, %v), want (model-a, nil)", model0, err)
				}
				if record0.SessionPolicy == nil ||
					record0.SessionPolicy["active_tool_loop"] != false ||
					record0.SessionPolicy["phase"] != string(selection.AgenticPhaseUserTurn) {
					t.Fatalf("turn 0 expected non-tool user_turn session_policy, got %+v", record0.SessionPolicy)
				}

				// Turn 1: tool_result + trailing system reminder must keep model-a as an active tool_loop.
				model1, ctx1, record1, err := runTurn(1, toolTurnVariant.body, candidates, "model-b")
				if err != nil || model1 != "model-a" {
					t.Fatalf("turn 1 switched model mid-tool-loop: got (%q, %v), want (model-a, nil)", model1, err)
				}
				if record1.SessionPolicy == nil {
					t.Fatal("turn 1 missing replay session_policy")
				}
				if record1.SessionPolicy["active_tool_loop"] != true {
					t.Fatalf("turn 1 expected active_tool_loop=true, got %+v", record1.SessionPolicy)
				}
				if record1.SessionPolicy["phase"] != string(selection.AgenticPhaseToolLoop) {
					t.Fatalf("turn 1 expected phase=%q, got %v (%+v)", selection.AgenticPhaseToolLoop, record1.SessionPolicy["phase"], record1.SessionPolicy)
				}
				if ctx1.VSRLearningProtectionPreflight == nil || ctx1.VSRLearningProtectionPreflight.Reason != "tool_or_protocol_state" {
					t.Fatalf("turn 1 expected preflight sampling suppression for tool_or_protocol_state, got %+v", ctx1.VSRLearningProtectionPreflight)
				}
				if scope == config.RouterLearningScopeConversation {
					if record1.SessionPolicy["hard_locked"] != true || record1.SessionPolicy["hard_lock_reason"] != "hard_lock=tool_loop" {
						t.Fatalf("turn 1 conversation scope expected hard_locked=true and hard_lock_reason=hard_lock=tool_loop, got %+v", record1.SessionPolicy)
					}
				} else if record1.SessionPolicy["reason"] != "session_scope_protect" {
					t.Fatalf("turn 1 session scope expected reason=session_scope_protect, got %+v", record1.SessionPolicy)
				}

				// If a decision's candidate set excludes the in-flight owner mid-tool-loop,
				// both scopes must fail closed rather than switching to model-b.
				if _, _, _, rejectErr := runTurn(2, toolTurnVariant.body, []config.ModelRef{{Model: "model-b"}}, "model-b"); !errors.Is(rejectErr, selection.ErrNoEligibleCandidates) {
					t.Fatalf("mid-tool-loop candidate mismatch error = %v, want ErrNoEligibleCandidates", rejectErr)
				}

				// Turn 3: completed tool loop followed by a new user turn + system reminder returns to user_turn.
				model3, _, record3, err := runTurn(3, turn3NextUserPromptAfterCompletedToolLoop, candidates, "model-b")
				if err != nil {
					t.Fatalf("turn 3 unexpected error: %v", err)
				}
				if record3.SessionPolicy == nil {
					t.Fatal("turn 3 missing replay session_policy")
				}
				if record3.SessionPolicy["active_tool_loop"] != false ||
					record3.SessionPolicy["hard_locked"] != false ||
					record3.SessionPolicy["phase"] != string(selection.AgenticPhaseUserTurn) {
					t.Fatalf("turn 3 after completed tool loop must return to unlocked user_turn, got %+v", record3.SessionPolicy)
				}
				wantTurn3Model := "model-b"
				if scope == config.RouterLearningScopeSession {
					wantTurn3Model = "model-a"
				}
				if model3 != wantTurn3Model {
					t.Fatalf("turn 3 selected %q under scope=%s, want %q", model3, scope, wantTurn3Model)
				}
			})
		}
	}
}
