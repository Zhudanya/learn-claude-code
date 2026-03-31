// s10 - Team Protocols (Go version)
// 关机握手 + 计划审批协议，基于 request_id 关联的 FSM。
// 运行: cd agents_go && go run ./s10/

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/uuid"
)

var (
	workdir   string
	modelID   string
	client    anthropic.Client
	systemMsg string
	teamDir   string
	inboxDir  string
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	systemMsg = fmt.Sprintf("You are a team lead at %s. Manage teammates with shutdown and plan approval protocols.", workdir)
	teamDir = filepath.Join(workdir, ".team")
	inboxDir = filepath.Join(teamDir, "inbox")
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// ========== Request Trackers ==========

type ShutdownRequest struct {
	Target string `json:"target"`
	Status string `json:"status"`
}

type PlanRequest struct {
	From   string `json:"from"`
	Plan   string `json:"plan"`
	Status string `json:"status"`
}

var (
	shutdownRequests = map[string]*ShutdownRequest{}
	planRequests     = map[string]*PlanRequest{}
	trackerLock      sync.Mutex
)

// ========== MessageBus ==========

type MessageBus struct {
	dir string
	mu  sync.Mutex
}

func NewMessageBus(dir string) *MessageBus {
	os.MkdirAll(dir, 0755)
	return &MessageBus{dir: dir}
}

type InboxMessage struct {
	Type      string  `json:"type"`
	From      string  `json:"from"`
	Content   string  `json:"content"`
	Timestamp float64 `json:"timestamp"`
	// 协议字段
	RequestID string `json:"request_id,omitempty"`
	Approve   *bool  `json:"approve,omitempty"`
	Plan      string `json:"plan,omitempty"`
	Feedback  string `json:"feedback,omitempty"`
}

func (mb *MessageBus) Send(sender, to, content, msgType string, extra map[string]any) string {
	if msgType == "" {
		msgType = "message"
	}
	msg := map[string]any{
		"type": msgType, "from": sender,
		"content": content, "timestamp": float64(time.Now().Unix()),
	}
	for k, v := range extra {
		msg[k] = v
	}
	mb.mu.Lock()
	defer mb.mu.Unlock()
	f, err := os.OpenFile(filepath.Join(mb.dir, to+".jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	defer f.Close()
	data, _ := json.Marshal(msg)
	f.Write(data)
	f.WriteString("\n")
	return fmt.Sprintf("Sent %s to %s", msgType, to)
}

func (mb *MessageBus) ReadInbox(name string) []map[string]any {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	path := filepath.Join(mb.dir, name+".jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var messages []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var msg map[string]any
		if json.Unmarshal([]byte(line), &msg) == nil {
			messages = append(messages, msg)
		}
	}
	os.WriteFile(path, []byte(""), 0644)
	return messages
}

func (mb *MessageBus) Broadcast(sender, content string, teammates []string) string {
	count := 0
	for _, name := range teammates {
		if name != sender {
			mb.Send(sender, name, content, "broadcast", nil)
			count++
		}
	}
	return fmt.Sprintf("Broadcast to %d teammates", count)
}

var bus *MessageBus

// ========== TeammateManager ==========

type TeamMember struct {
	Name   string `json:"name"`
	Role   string `json:"role"`
	Status string `json:"status"`
}

type TeamConfig struct {
	TeamName string       `json:"team_name"`
	Members  []TeamMember `json:"members"`
}

type TeammateManager struct {
	dir        string
	configPath string
	config     TeamConfig
	mu         sync.Mutex
}

func NewTeammateManager(dir string) *TeammateManager {
	os.MkdirAll(dir, 0755)
	tm := &TeammateManager{dir: dir, configPath: filepath.Join(dir, "config.json")}
	tm.config = tm.loadConfig()
	return tm
}

func (tm *TeammateManager) loadConfig() TeamConfig {
	data, err := os.ReadFile(tm.configPath)
	if err != nil {
		return TeamConfig{TeamName: "default"}
	}
	var cfg TeamConfig
	json.Unmarshal(data, &cfg)
	return cfg
}

func (tm *TeammateManager) saveConfig() {
	data, _ := json.MarshalIndent(tm.config, "", "  ")
	os.WriteFile(tm.configPath, data, 0644)
}

func (tm *TeammateManager) findMember(name string) *TeamMember {
	for i := range tm.config.Members {
		if tm.config.Members[i].Name == name {
			return &tm.config.Members[i]
		}
	}
	return nil
}

func (tm *TeammateManager) Spawn(name, role, prompt string) string {
	tm.mu.Lock()
	member := tm.findMember(name)
	if member != nil {
		if member.Status != "idle" && member.Status != "shutdown" {
			tm.mu.Unlock()
			return fmt.Sprintf("Error: '%s' is currently %s", name, member.Status)
		}
		member.Status = "working"
		member.Role = role
	} else {
		tm.config.Members = append(tm.config.Members, TeamMember{Name: name, Role: role, Status: "working"})
	}
	tm.saveConfig()
	tm.mu.Unlock()

	go tm.teammateLoop(name, role, prompt)
	return fmt.Sprintf("Spawned '%s' (role: %s)", name, role)
}

func (tm *TeammateManager) teammateLoop(name, role, prompt string) {
	sysPrompt := fmt.Sprintf(
		"You are '%s', role: %s, at %s. Submit plans via plan_approval before major work. Respond to shutdown_request with shutdown_response.",
		name, role, workdir)
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
	}
	tools := tm.getTeammateTools()
	ctx := context.Background()
	shouldExit := false

	for i := 0; i < 50; i++ {
		inbox := bus.ReadInbox(name)
		for _, msg := range inbox {
			data, _ := json.Marshal(msg)
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(string(data))))
		}
		if shouldExit {
			break
		}

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model: anthropic.Model(modelID), System: []anthropic.TextBlockParam{{Text: sysPrompt}},
			Messages: messages, Tools: tools, MaxTokens: 8000,
		})
		if err != nil {
			break
		}

		var blocks []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			switch b.Type {
			case "text":
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfText: &anthropic.TextBlockParam{Text: b.AsText().Text}})
			case "tool_use":
				tu := b.AsToolUse()
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolUse: &anthropic.ToolUseBlockParam{ID: tu.ID, Name: tu.Name, Input: tu.Input}})
			}
		}
		messages = append(messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleAssistant, Content: blocks})

		if resp.StopReason != anthropic.StopReasonToolUse {
			break
		}

		var results []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			if b.Type == "tool_use" {
				tu := b.AsToolUse()
				output := tm.execTool(name, tu.Name, tu.Input)
				fmt.Printf("  [%s] %s: %s\n", name, tu.Name, truncate(output, 120))

				// 检查是否同意关机
				if tu.Name == "shutdown_response" {
					var p map[string]any
					json.Unmarshal(tu.Input, &p)
					if approve, ok := p["approve"].(bool); ok && approve {
						shouldExit = true
					}
				}

				results = append(results, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: tu.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}}}})
			}
		}
		messages = append(messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: results})
	}

	tm.mu.Lock()
	if m := tm.findMember(name); m != nil {
		if shouldExit {
			m.Status = "shutdown"
		} else {
			m.Status = "idle"
		}
		tm.saveConfig()
	}
	tm.mu.Unlock()
}

func (tm *TeammateManager) execTool(sender, toolName string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	boolVal := func(k string) bool { v, _ := p[k].(bool); return v }

	switch toolName {
	case "bash":
		return runBash(str("command"))
	case "read_file":
		return runRead(str("path"), 0)
	case "write_file":
		return runWrite(str("path"), str("content"))
	case "edit_file":
		return runEdit(str("path"), str("old_text"), str("new_text"))
	case "send_message":
		msgType := str("msg_type")
		if msgType == "" {
			msgType = "message"
		}
		return bus.Send(sender, str("to"), str("content"), msgType, nil)
	case "read_inbox":
		msgs := bus.ReadInbox(sender)
		data, _ := json.MarshalIndent(msgs, "", "  ")
		return string(data)
	case "shutdown_response":
		reqID := str("request_id")
		approve := boolVal("approve")
		trackerLock.Lock()
		if req, ok := shutdownRequests[reqID]; ok {
			if approve {
				req.Status = "approved"
			} else {
				req.Status = "rejected"
			}
		}
		trackerLock.Unlock()
		bus.Send(sender, "lead", str("reason"), "shutdown_response",
			map[string]any{"request_id": reqID, "approve": approve})
		if approve {
			return "Shutdown approved"
		}
		return "Shutdown rejected"
	case "plan_approval":
		planText := str("plan")
		reqID := uuid.New().String()[:8]
		trackerLock.Lock()
		planRequests[reqID] = &PlanRequest{From: sender, Plan: planText, Status: "pending"}
		trackerLock.Unlock()
		bus.Send(sender, "lead", planText, "plan_approval_response",
			map[string]any{"request_id": reqID, "plan": planText})
		return fmt.Sprintf("Plan submitted (request_id=%s). Waiting for lead approval.", reqID)
	default:
		return fmt.Sprintf("Unknown tool: %s", toolName)
	}
}

func (tm *TeammateManager) getTeammateTools() []anthropic.ToolUnionParam {
	return []anthropic.ToolUnionParam{
		{OfTool: &anthropic.ToolParam{Name: "bash", Description: anthropic.String("Run a shell command."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"command": map[string]any{"type": "string"}}, Required: []string{"command"}}}},
		{OfTool: &anthropic.ToolParam{Name: "read_file", Description: anthropic.String("Read file contents."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}}, Required: []string{"path"}}}},
		{OfTool: &anthropic.ToolParam{Name: "write_file", Description: anthropic.String("Write content to file."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}}, Required: []string{"path", "content"}}}},
		{OfTool: &anthropic.ToolParam{Name: "edit_file", Description: anthropic.String("Replace exact text in file."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "old_text": map[string]any{"type": "string"}, "new_text": map[string]any{"type": "string"}}, Required: []string{"path", "old_text", "new_text"}}}},
		{OfTool: &anthropic.ToolParam{Name: "send_message", Description: anthropic.String("Send message to a teammate."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"to": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}, "msg_type": map[string]any{"type": "string"}}, Required: []string{"to", "content"}}}},
		{OfTool: &anthropic.ToolParam{Name: "read_inbox", Description: anthropic.String("Read and drain your inbox."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
		{OfTool: &anthropic.ToolParam{Name: "shutdown_response", Description: anthropic.String("Respond to a shutdown request. Approve to shut down, reject to keep working."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}, "approve": map[string]any{"type": "boolean"}, "reason": map[string]any{"type": "string"}}, Required: []string{"request_id", "approve"}}}},
		{OfTool: &anthropic.ToolParam{Name: "plan_approval", Description: anthropic.String("Submit a plan for lead approval."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"plan": map[string]any{"type": "string"}}, Required: []string{"plan"}}}},
	}
}

func (tm *TeammateManager) ListAll() string {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if len(tm.config.Members) == 0 {
		return "No teammates."
	}
	lines := []string{fmt.Sprintf("Team: %s", tm.config.TeamName)}
	for _, m := range tm.config.Members {
		lines = append(lines, fmt.Sprintf("  %s (%s): %s", m.Name, m.Role, m.Status))
	}
	return strings.Join(lines, "\n")
}

func (tm *TeammateManager) MemberNames() []string {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	var names []string
	for _, m := range tm.config.Members {
		names = append(names, m.Name)
	}
	return names
}

var team *TeammateManager

// ========== 基础工具 ==========

func safePath(p string) (string, error) {
	abs, err := filepath.Abs(filepath.Join(workdir, p))
	if err != nil {
		return "", err
	}
	absW, _ := filepath.Abs(workdir)
	if !strings.HasPrefix(abs, absW) {
		return "", fmt.Errorf("path escapes workspace: %s", p)
	}
	return abs, nil
}

func runBash(command string) string {
	for _, d := range []string{"rm -rf /", "sudo", "shutdown", "reboot"} {
		if strings.Contains(command, d) {
			return "Error: Dangerous command blocked"
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "Error: Timeout (120s)"
	}
	result := strings.TrimSpace(string(out))
	if err != nil && result == "" {
		return fmt.Sprintf("Error: %v", err)
	}
	if result == "" {
		return "(no output)"
	}
	return truncate(result, 50000)
}

func runRead(path string, limit int) string {
	fp, err := safePath(path)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	data, err := os.ReadFile(fp)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	lines := strings.Split(string(data), "\n")
	if limit > 0 && limit < len(lines) {
		lines = append(lines[:limit], fmt.Sprintf("... (%d more)", len(lines)-limit))
	}
	return truncate(strings.Join(lines, "\n"), 50000)
}

func runWrite(path, content string) string {
	fp, err := safePath(path)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	os.MkdirAll(filepath.Dir(fp), 0755)
	if err := os.WriteFile(fp, []byte(content), 0644); err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return fmt.Sprintf("Wrote %d bytes", len(content))
}

func runEdit(path, oldText, newText string) string {
	fp, err := safePath(path)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	data, err := os.ReadFile(fp)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	c := string(data)
	if !strings.Contains(c, oldText) {
		return fmt.Sprintf("Error: Text not found in %s", path)
	}
	os.WriteFile(fp, []byte(strings.Replace(c, oldText, newText, 1)), 0644)
	return fmt.Sprintf("Edited %s", path)
}

// ========== 领导协议处理函数 ==========

func handleShutdownRequest(teammate string) string {
	reqID := uuid.New().String()[:8]
	trackerLock.Lock()
	shutdownRequests[reqID] = &ShutdownRequest{Target: teammate, Status: "pending"}
	trackerLock.Unlock()
	bus.Send("lead", teammate, "Please shut down gracefully.", "shutdown_request",
		map[string]any{"request_id": reqID})
	return fmt.Sprintf("Shutdown request %s sent to '%s' (status: pending)", reqID, teammate)
}

func handlePlanReview(requestID string, approve bool, feedback string) string {
	trackerLock.Lock()
	req, ok := planRequests[requestID]
	if !ok {
		trackerLock.Unlock()
		return fmt.Sprintf("Error: Unknown plan request_id '%s'", requestID)
	}
	if approve {
		req.Status = "approved"
	} else {
		req.Status = "rejected"
	}
	from := req.From
	trackerLock.Unlock()
	bus.Send("lead", from, feedback, "plan_approval_response",
		map[string]any{"request_id": requestID, "approve": approve, "feedback": feedback})
	return fmt.Sprintf("Plan %s for '%s'", req.Status, from)
}

func checkShutdownStatus(requestID string) string {
	trackerLock.Lock()
	defer trackerLock.Unlock()
	req, ok := shutdownRequests[requestID]
	if !ok {
		return `{"error": "not found"}`
	}
	data, _ := json.Marshal(req)
	return string(data)
}

// ========== 领导工具分发（12 个） ==========

func dispatchTool(name string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	boolVal := func(k string) bool { v, _ := p[k].(bool); return v }

	switch name {
	case "bash":
		return runBash(str("command"))
	case "read_file":
		return runRead(str("path"), 0)
	case "write_file":
		return runWrite(str("path"), str("content"))
	case "edit_file":
		return runEdit(str("path"), str("old_text"), str("new_text"))
	case "spawn_teammate":
		return team.Spawn(str("name"), str("role"), str("prompt"))
	case "list_teammates":
		return team.ListAll()
	case "send_message":
		msgType := str("msg_type")
		if msgType == "" {
			msgType = "message"
		}
		return bus.Send("lead", str("to"), str("content"), msgType, nil)
	case "read_inbox":
		msgs := bus.ReadInbox("lead")
		data, _ := json.MarshalIndent(msgs, "", "  ")
		return string(data)
	case "broadcast":
		return bus.Broadcast("lead", str("content"), team.MemberNames())
	case "shutdown_request":
		return handleShutdownRequest(str("teammate"))
	case "shutdown_response":
		return checkShutdownStatus(str("request_id"))
	case "plan_approval":
		return handlePlanReview(str("request_id"), boolVal("approve"), str("feedback"))
	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

var leadTools = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{Name: "bash", Description: anthropic.String("Run a shell command."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"command": map[string]any{"type": "string"}}, Required: []string{"command"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_file", Description: anthropic.String("Read file contents."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, Required: []string{"path"}}}},
	{OfTool: &anthropic.ToolParam{Name: "write_file", Description: anthropic.String("Write content to file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}}, Required: []string{"path", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "edit_file", Description: anthropic.String("Replace exact text in file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "old_text": map[string]any{"type": "string"}, "new_text": map[string]any{"type": "string"}}, Required: []string{"path", "old_text", "new_text"}}}},
	{OfTool: &anthropic.ToolParam{Name: "spawn_teammate", Description: anthropic.String("Spawn a persistent teammate."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}, "role": map[string]any{"type": "string"}, "prompt": map[string]any{"type": "string"}}, Required: []string{"name", "role", "prompt"}}}},
	{OfTool: &anthropic.ToolParam{Name: "list_teammates", Description: anthropic.String("List all teammates."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "send_message", Description: anthropic.String("Send a message to a teammate."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"to": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}, "msg_type": map[string]any{"type": "string"}}, Required: []string{"to", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_inbox", Description: anthropic.String("Read and drain the lead's inbox."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "broadcast", Description: anthropic.String("Send a message to all teammates."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"content": map[string]any{"type": "string"}}, Required: []string{"content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "shutdown_request", Description: anthropic.String("Request a teammate to shut down gracefully."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"teammate": map[string]any{"type": "string"}}, Required: []string{"teammate"}}}},
	{OfTool: &anthropic.ToolParam{Name: "shutdown_response", Description: anthropic.String("Check the status of a shutdown request by request_id."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}}, Required: []string{"request_id"}}}},
	{OfTool: &anthropic.ToolParam{Name: "plan_approval", Description: anthropic.String("Approve or reject a teammate's plan."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}, "approve": map[string]any{"type": "boolean"}, "feedback": map[string]any{"type": "string"}}, Required: []string{"request_id", "approve"}}}},
}

// ========== 领导 Agent Loop ==========

func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	for {
		inbox := bus.ReadInbox("lead")
		if len(inbox) > 0 {
			data, _ := json.MarshalIndent(inbox, "", "  ")
			*messages = append(*messages,
				anthropic.NewUserMessage(anthropic.NewTextBlock(fmt.Sprintf("<inbox>%s</inbox>", string(data)))),
				anthropic.NewAssistantMessage(anthropic.NewTextBlock("Noted inbox messages.")),
			)
		}

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model: anthropic.Model(modelID), System: []anthropic.TextBlockParam{{Text: systemMsg}},
			Messages: *messages, Tools: leadTools, MaxTokens: 8000,
		})
		if err != nil {
			return err
		}

		var blocks []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			switch b.Type {
			case "text":
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfText: &anthropic.TextBlockParam{Text: b.AsText().Text}})
			case "tool_use":
				tu := b.AsToolUse()
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolUse: &anthropic.ToolUseBlockParam{ID: tu.ID, Name: tu.Name, Input: tu.Input}})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleAssistant, Content: blocks})

		if resp.StopReason != anthropic.StopReasonToolUse {
			return nil
		}

		var results []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			if b.Type == "tool_use" {
				tu := b.AsToolUse()
				output := dispatchTool(tu.Name, tu.Input)
				fmt.Printf("> %s: %s\n", tu.Name, truncate(output, 200))
				results = append(results, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: tu.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}}}})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: results})
	}
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY not set")
		os.Exit(1)
	}

	bus = NewMessageBus(inboxDir)
	team = NewTeammateManager(teamDir)

	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\033[36ms10-go >> \033[0m")
		if !scanner.Scan() {
			break
		}
		query := strings.TrimSpace(scanner.Text())
		if query == "" || query == "q" || query == "exit" {
			break
		}
		if query == "/team" {
			fmt.Println(team.ListAll())
			continue
		}
		if query == "/inbox" {
			msgs := bus.ReadInbox("lead")
			data, _ := json.MarshalIndent(msgs, "", "  ")
			fmt.Println(string(data))
			continue
		}

		history = append(history, anthropic.NewUserMessage(anthropic.NewTextBlock(query)))
		if err := agentLoop(&history); err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		if len(history) > 0 {
			for _, b := range history[len(history)-1].Content {
				if b.OfText != nil {
					fmt.Println(b.OfText.Text)
				}
			}
		}
		fmt.Println()
	}
}
