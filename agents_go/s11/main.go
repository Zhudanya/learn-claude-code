// s11 - Autonomous Agents (Go version)
// 自治队友：WORK + IDLE 两阶段循环，自动扫描任务看板认领，身份重注入。
// 运行: cd agents_go && go run ./s11/

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
	"sort"
	"strconv"
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
	tasksDir  string
)

const (
	pollInterval = 5  // 秒
	idleTimeout  = 60 // 秒
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	systemMsg = fmt.Sprintf("You are a team lead at %s. Teammates are autonomous -- they find work themselves.", workdir)
	teamDir = filepath.Join(workdir, ".team")
	inboxDir = filepath.Join(teamDir, "inbox")
	tasksDir = filepath.Join(workdir, ".tasks")
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

// ========== Trackers + Locks ==========

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
	claimLock        sync.Mutex
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
	f, _ := os.OpenFile(filepath.Join(mb.dir, to+".jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
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
	var msgs []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var msg map[string]any
		if json.Unmarshal([]byte(line), &msg) == nil {
			msgs = append(msgs, msg)
		}
	}
	os.WriteFile(path, []byte(""), 0644)
	return msgs
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

// ========== Task Board ==========

type TaskData struct {
	ID          int    `json:"id"`
	Subject     string `json:"subject"`
	Description string `json:"description"`
	Status      string `json:"status"`
	Owner       string `json:"owner"`
	BlockedBy   []int  `json:"blockedBy"`
	Blocks      []int  `json:"blocks"`
}

func scanUnclaimedTasks() []TaskData {
	os.MkdirAll(tasksDir, 0755)
	entries, _ := filepath.Glob(filepath.Join(tasksDir, "task_*.json"))
	sort.Strings(entries)
	var unclaimed []TaskData
	for _, e := range entries {
		data, _ := os.ReadFile(e)
		var t TaskData
		json.Unmarshal(data, &t)
		if t.Status == "pending" && t.Owner == "" && len(t.BlockedBy) == 0 {
			unclaimed = append(unclaimed, t)
		}
	}
	return unclaimed
}

func claimTask(taskID int, owner string) string {
	claimLock.Lock()
	defer claimLock.Unlock()
	path := filepath.Join(tasksDir, fmt.Sprintf("task_%d.json", taskID))
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("Error: Task %d not found", taskID)
	}
	var t TaskData
	json.Unmarshal(data, &t)
	t.Owner = owner
	t.Status = "in_progress"
	out, _ := json.MarshalIndent(t, "", "  ")
	os.WriteFile(path, out, 0644)
	return fmt.Sprintf("Claimed task #%d for %s", taskID, owner)
}

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

func (tm *TeammateManager) setStatus(name, status string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if m := tm.findMember(name); m != nil {
		m.Status = status
		tm.saveConfig()
	}
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

	go tm.loop(name, role, prompt)
	return fmt.Sprintf("Spawned '%s' (role: %s)", name, role)
}

func (tm *TeammateManager) loop(name, role, prompt string) {
	teamName := tm.config.TeamName
	sysPrompt := fmt.Sprintf(
		"You are '%s', role: %s, team: %s, at %s. Use idle tool when you have no more work. You will auto-claim new tasks.",
		name, role, teamName, workdir)
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
	}
	tools := tm.getTeammateTools()
	ctx := context.Background()

	for { // 外层 while True
		// -- WORK PHASE --
		for i := 0; i < 50; i++ {
			inbox := bus.ReadInbox(name)
			for _, msg := range inbox {
				if msgType, _ := msg["type"].(string); msgType == "shutdown_request" {
					tm.setStatus(name, "shutdown")
					return
				}
				data, _ := json.Marshal(msg)
				messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(string(data))))
			}

			resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
				Model: anthropic.Model(modelID), System: []anthropic.TextBlockParam{{Text: sysPrompt}},
				Messages: messages, Tools: tools, MaxTokens: 8000,
			})
			if err != nil {
				tm.setStatus(name, "idle")
				return
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
			idleRequested := false
			for _, b := range resp.Content {
				if b.Type == "tool_use" {
					tu := b.AsToolUse()
					var output string
					if tu.Name == "idle" {
						idleRequested = true
						output = "Entering idle phase. Will poll for new tasks."
					} else {
						output = tm.execTool(name, tu.Name, tu.Input)
					}
					fmt.Printf("  [%s] %s: %s\n", name, tu.Name, truncate(output, 120))
					results = append(results, anthropic.ContentBlockParamUnion{
						OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: tu.ID,
							Content: []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}}}})
				}
			}
			messages = append(messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: results})
			if idleRequested {
				break
			}
		}

		// -- IDLE PHASE --
		tm.setStatus(name, "idle")
		resume := false
		polls := idleTimeout / pollInterval
		for p := 0; p < polls; p++ {
			time.Sleep(time.Duration(pollInterval) * time.Second)

			// 1. 检查收件箱
			inbox := bus.ReadInbox(name)
			if len(inbox) > 0 {
				for _, msg := range inbox {
					if msgType, _ := msg["type"].(string); msgType == "shutdown_request" {
						tm.setStatus(name, "shutdown")
						return
					}
					data, _ := json.Marshal(msg)
					messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(string(data))))
				}
				resume = true
				break
			}

			// 2. 扫描任务看板
			unclaimed := scanUnclaimedTasks()
			if len(unclaimed) > 0 {
				task := unclaimed[0]
				claimTask(task.ID, name)
				taskPrompt := fmt.Sprintf("<auto-claimed>Task #%d: %s\n%s</auto-claimed>", task.ID, task.Subject, task.Description)
				// 身份重注入
				if len(messages) <= 3 {
					messages = append(
						[]anthropic.MessageParam{
							anthropic.NewUserMessage(anthropic.NewTextBlock(
								fmt.Sprintf("<identity>You are '%s', role: %s, team: %s. Continue your work.</identity>", name, role, teamName))),
							anthropic.NewAssistantMessage(anthropic.NewTextBlock(fmt.Sprintf("I am %s. Continuing.", name))),
						},
						messages...,
					)
				}
				messages = append(messages,
					anthropic.NewUserMessage(anthropic.NewTextBlock(taskPrompt)),
					anthropic.NewAssistantMessage(anthropic.NewTextBlock(fmt.Sprintf("Claimed task #%d. Working on it.", task.ID))),
				)
				resume = true
				break
			}
		}

		if !resume {
			tm.setStatus(name, "shutdown")
			return
		}
		tm.setStatus(name, "working")
	}
}

func (tm *TeammateManager) execTool(sender, toolName string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	boolVal := func(k string) bool { v, _ := p[k].(bool); return v }
	intVal := func(k string) int { v, _ := p[k].(float64); return int(v) }

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
		mt := str("msg_type")
		if mt == "" {
			mt = "message"
		}
		return bus.Send(sender, str("to"), str("content"), mt, nil)
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
		reqID := uuid.New().String()[:8]
		planText := str("plan")
		trackerLock.Lock()
		planRequests[reqID] = &PlanRequest{From: sender, Plan: planText, Status: "pending"}
		trackerLock.Unlock()
		bus.Send(sender, "lead", planText, "plan_approval_response",
			map[string]any{"request_id": reqID, "plan": planText})
		return fmt.Sprintf("Plan submitted (request_id=%s). Waiting for approval.", reqID)
	case "claim_task":
		return claimTask(intVal("task_id"), sender)
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
		{OfTool: &anthropic.ToolParam{Name: "shutdown_response", Description: anthropic.String("Respond to a shutdown request."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}, "approve": map[string]any{"type": "boolean"}, "reason": map[string]any{"type": "string"}}, Required: []string{"request_id", "approve"}}}},
		{OfTool: &anthropic.ToolParam{Name: "plan_approval", Description: anthropic.String("Submit a plan for lead approval."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"plan": map[string]any{"type": "string"}}, Required: []string{"plan"}}}},
		{OfTool: &anthropic.ToolParam{Name: "idle", Description: anthropic.String("Signal that you have no more work. Enters idle polling phase."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
		{OfTool: &anthropic.ToolParam{Name: "claim_task", Description: anthropic.String("Claim a task from the board by ID."),
			InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}}, Required: []string{"task_id"}}}},
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

// ========== 领导协议处理 ==========

func handleShutdownRequest(teammate string) string {
	reqID := uuid.New().String()[:8]
	trackerLock.Lock()
	shutdownRequests[reqID] = &ShutdownRequest{Target: teammate, Status: "pending"}
	trackerLock.Unlock()
	bus.Send("lead", teammate, "Please shut down gracefully.", "shutdown_request",
		map[string]any{"request_id": reqID})
	return fmt.Sprintf("Shutdown request %s sent to '%s'", reqID, teammate)
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

// ========== 领导工具分发（14 个） ==========

func dispatchTool(name string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	boolVal := func(k string) bool { v, _ := p[k].(bool); return v }
	intVal := func(k string) int { v, _ := p[k].(float64); return int(v) }

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
		mt := str("msg_type")
		if mt == "" {
			mt = "message"
		}
		return bus.Send("lead", str("to"), str("content"), mt, nil)
	case "read_inbox":
		msgs := bus.ReadInbox("lead")
		data, _ := json.MarshalIndent(msgs, "", "  ")
		return string(data)
	case "broadcast":
		return bus.Broadcast("lead", str("content"), team.MemberNames())
	case "shutdown_request":
		return handleShutdownRequest(str("teammate"))
	case "shutdown_response":
		trackerLock.Lock()
		req, ok := shutdownRequests[str("request_id")]
		trackerLock.Unlock()
		if !ok {
			return `{"error":"not found"}`
		}
		data, _ := json.Marshal(req)
		return string(data)
	case "plan_approval":
		return handlePlanReview(str("request_id"), boolVal("approve"), str("feedback"))
	case "idle":
		return "Lead does not idle."
	case "claim_task":
		return claimTask(intVal("task_id"), "lead")
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
	{OfTool: &anthropic.ToolParam{Name: "spawn_teammate", Description: anthropic.String("Spawn an autonomous teammate."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}, "role": map[string]any{"type": "string"}, "prompt": map[string]any{"type": "string"}}, Required: []string{"name", "role", "prompt"}}}},
	{OfTool: &anthropic.ToolParam{Name: "list_teammates", Description: anthropic.String("List all teammates."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "send_message", Description: anthropic.String("Send a message to a teammate."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"to": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}, "msg_type": map[string]any{"type": "string"}}, Required: []string{"to", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_inbox", Description: anthropic.String("Read and drain the lead's inbox."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "broadcast", Description: anthropic.String("Send a message to all teammates."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"content": map[string]any{"type": "string"}}, Required: []string{"content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "shutdown_request", Description: anthropic.String("Request a teammate to shut down."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"teammate": map[string]any{"type": "string"}}, Required: []string{"teammate"}}}},
	{OfTool: &anthropic.ToolParam{Name: "shutdown_response", Description: anthropic.String("Check shutdown request status."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}}, Required: []string{"request_id"}}}},
	{OfTool: &anthropic.ToolParam{Name: "plan_approval", Description: anthropic.String("Approve or reject a teammate's plan."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"request_id": map[string]any{"type": "string"}, "approve": map[string]any{"type": "boolean"}, "feedback": map[string]any{"type": "string"}}, Required: []string{"request_id", "approve"}}}},
	{OfTool: &anthropic.ToolParam{Name: "idle", Description: anthropic.String("Enter idle state (for lead -- rarely used)."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "claim_task", Description: anthropic.String("Claim a task from the board by ID."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}}, Required: []string{"task_id"}}}},
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

// ========== 主程序 ==========

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
		fmt.Print("\033[36ms11-go >> \033[0m")
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
		if query == "/tasks" {
			os.MkdirAll(tasksDir, 0755)
			entries, _ := filepath.Glob(filepath.Join(tasksDir, "task_*.json"))
			sort.Strings(entries)
			for _, e := range entries {
				data, _ := os.ReadFile(e)
				var t TaskData
				json.Unmarshal(data, &t)
				markers := map[string]string{"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
				m := markers[t.Status]
				if m == "" {
					m = "[?]"
				}
				owner := ""
				if t.Owner != "" {
					owner = " @" + t.Owner
				}
				fmt.Printf("  %s #%d: %s%s\n", m, t.ID, t.Subject, owner)
			}
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

// unused import guard
var _ = strconv.Itoa
