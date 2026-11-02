package extproc

import "testing"

func TestExtractContentFastMetadataAndImageCount(t *testing.T) {
	body := []byte(`{
		"model":"vllm-sr/test",
		"metadata":{"consent":"denied","cohort":"canary"},
		"messages":[{"role":"user","content":[
			{"type":"text","text":"hello"},
			{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
		]}]
	}`)

	result, err := extractContentFast(body)
	if err != nil {
		t.Fatalf("extractContentFast() error = %v", err)
	}
	if result.Metadata["consent"] != "denied" {
		t.Fatalf("metadata = %v", result.Metadata)
	}
	if result.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1", result.ImageContentCount)
	}
	if result.FirstImageURL != "" {
		t.Fatalf("FirstImageURL = %q, remote URL should remain unavailable to embeddings", result.FirstImageURL)
	}
}

func TestExtractContentFastRejectsNonStringMetadata(t *testing.T) {
	body := []byte(`{"model":"m","metadata":{"cohort":1},"messages":[]}`)
	if _, err := extractContentFast(body); err == nil {
		t.Fatal("extractContentFast() expected metadata type error")
	}
}

func TestExtractContentFastRejectsNonStringMetadataWithEmptyKey(t *testing.T) {
	body := []byte(`{"model":"m","metadata":{"":1},"messages":[]}`)
	if _, err := extractContentFast(body); err == nil {
		t.Fatal("extractContentFast() expected empty-key metadata type error")
	}
}

func TestExtractContentFastPreservesToolLoopAcrossSystemAndDeveloperMessages(t *testing.T) {
	tests := []struct {
		name                string
		body                string
		wantLastRole        string
		wantSystemAfterTool bool
		wantUserAfterTool   bool
	}{
		{
			name: "tool_then_system",
			body: `{"model":"m","messages":[
				{"role":"user","content":"Run uname -m"},
				{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"aarch64"},
				{"role":"system","content":"<system-reminder><total_tokens>14984549</total_tokens></system-reminder>"}
			]}`,
			wantLastRole:        "system",
			wantSystemAfterTool: true,
		},
		{
			name: "tool_then_developer_then_user",
			body: `{"model":"m","messages":[
				{"role":"user","content":"Run uname -m"},
				{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"aarch64"},
				{"role":"developer","content":"# Environment update"},
				{"role":"user","content":"Now fix it"}
			]}`,
			wantLastRole:      "user",
			wantUserAfterTool: true,
		},
		{
			name: "tool_then_user_then_system",
			body: `{"model":"m","messages":[
				{"role":"user","content":"Run uname -m"},
				{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"aarch64"},
				{"role":"user","content":"<system-reminder>file modified</system-reminder>"},
				{"role":"system","content":"<system-reminder><total_tokens>14984549</total_tokens></system-reminder>"}
			]}`,
			wantLastRole:      "system",
			wantUserAfterTool: true,
		},
		{
			name: "user_then_system_without_tool",
			body: `{"model":"m","messages":[
				{"role":"user","content":"Run uname -m"},
				{"role":"system","content":"<system-reminder>tokens</system-reminder>"}
			]}`,
			wantLastRole: "system",
		},
		{
			name: "completed_run_assistant_then_system",
			body: `{"model":"m","messages":[
				{"role":"user","content":"Run uname -m"},
				{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"call_1","content":"aarch64"},
				{"role":"system","content":"<system-reminder>tokens</system-reminder>"},
				{"role":"assistant","content":"Done."},
				{"role":"system","content":"<system-reminder>tokens</system-reminder>"}
			]}`,
			wantLastRole: "system",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := extractContentFast([]byte(tc.body))
			if err != nil {
				t.Fatalf("extractContentFast() error = %v", err)
			}
			if result.LastMessageRole != tc.wantLastRole {
				t.Fatalf("LastMessageRole = %q, want %q", result.LastMessageRole, tc.wantLastRole)
			}
			if result.LastMessageToolResult {
				t.Fatal("LastMessageToolResult = true, want false")
			}
			if result.LastSystemAfterToolResult != tc.wantSystemAfterTool {
				t.Fatalf("LastSystemAfterToolResult = %v, want %v", result.LastSystemAfterToolResult, tc.wantSystemAfterTool)
			}
			if result.LastUserAfterToolResult != tc.wantUserAfterTool {
				t.Fatalf("LastUserAfterToolResult = %v, want %v", result.LastUserAfterToolResult, tc.wantUserAfterTool)
			}
		})
	}
}
