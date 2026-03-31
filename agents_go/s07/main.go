// s07 - Task System (Go version)
// 持久化任务图：JSON 文件 + 依赖关系（blockedBy/blocks）。
// 运行: cd agents_go && go run ./s07/

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
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

var (
	workdir  string
	modelID  string
	client   anthropic.Client
	systemMsg string
	tasksDir string
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	systemMsg = fmt.Sprintf("You are a coding agent at %s. Use task tools to plan and track work.", workdir)
	tasksDir = filepath.Join(workdir, ".tasks")
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

// ========== TaskManager ==========

type Task struct {
	ID          int    `json:"id"`
	Subject     string `json:"subject"`
	Description string `json:"description"`
	Status      string `json:"status"`
	BlockedBy   []int  `json:"blockedBy"`
	Blocks      []int  `json:"blocks"`
	Owner       string `json:"owner"`
}

type TaskManager struct {
	dir    string
	nextID int
}

func NewTaskManager(dir string) *TaskManager {
	os.MkdirAll(dir, 0755)
	tm := &TaskManager{dir: dir}
	tm.nextID = tm.maxID() + 1
	return tm
}

func (tm *TaskManager) maxID() int {
	entries, _ := filepath.Glob(filepath.Join(tm.dir, "task_*.json"))
	maxVal := 0
	for _, e := range entries {
		base := filepath.Base(e)
		numStr := strings.TrimSuffix(strings.TrimPrefix(base, "task_"), ".json")
		if n, err := strconv.Atoi(numStr); err == nil && n > maxVal {
			maxVal = n
		}
	}
	return maxVal
}

func (tm *TaskManager) load(taskID int) (*Task, error) {
	path := filepath.Join(tm.dir, fmt.Sprintf("task_%d.json", taskID))
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("task %d not found", taskID)
	}
	var t Task
	json.Unmarshal(data, &t)
	return &t, nil
}

func (tm *TaskManager) save(t *Task) {
	data, _ := json.MarshalIndent(t, "", "  ")
	os.WriteFile(filepath.Join(tm.dir, fmt.Sprintf("task_%d.json", t.ID)), data, 0644)
}

func (tm *TaskManager) Create(subject, description string) string {
	t := &Task{ID: tm.nextID, Subject: subject, Description: description, Status: "pending"}
	tm.save(t)
	tm.nextID++
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
}

func (tm *TaskManager) Get(taskID int) string {
	t, err := tm.load(taskID)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
}

func (tm *TaskManager) Update(taskID int, status string, addBlockedBy, addBlocks []int) string {
	t, err := tm.load(taskID)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	if status != "" {
		if status != "pending" && status != "in_progress" && status != "completed" {
			return fmt.Sprintf("Error: invalid status '%s'", status)
		}
		t.Status = status
		if status == "completed" {
			tm.clearDependency(taskID)
		}
	}
	if len(addBlockedBy) > 0 {
		t.BlockedBy = uniqueAppend(t.BlockedBy, addBlockedBy)
	}
	if len(addBlocks) > 0 {
		t.Blocks = uniqueAppend(t.Blocks, addBlocks)
		for _, blockedID := range addBlocks {
			if blocked, err := tm.load(blockedID); err == nil {
				if !contains(blocked.BlockedBy, taskID) {
					blocked.BlockedBy = append(blocked.BlockedBy, taskID)
					tm.save(blocked)
				}
			}
		}
	}
	tm.save(t)
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
}

func (tm *TaskManager) clearDependency(completedID int) {
	entries, _ := filepath.Glob(filepath.Join(tm.dir, "task_*.json"))
	for _, e := range entries {
		data, _ := os.ReadFile(e)
		var t Task
		json.Unmarshal(data, &t)
		if contains(t.BlockedBy, completedID) {
			t.BlockedBy = remove(t.BlockedBy, completedID)
			tm.save(&t)
		}
	}
}

func (tm *TaskManager) ListAll() string {
	entries, _ := filepath.Glob(filepath.Join(tm.dir, "task_*.json"))
	if len(entries) == 0 {
		return "No tasks."
	}
	sort.Strings(entries)
	var lines []string
	for _, e := range entries {
		data, _ := os.ReadFile(e)
		var t Task
		json.Unmarshal(data, &t)
		markers := map[string]string{"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
		m := markers[t.Status]
		if m == "" {
			m = "[?]"
		}
		blocked := ""
		if len(t.BlockedBy) > 0 {
			blocked = fmt.Sprintf(" (blocked by: %v)", t.BlockedBy)
		}
		lines = append(lines, fmt.Sprintf("%s #%d: %s%s", m, t.ID, t.Subject, blocked))
	}
	return strings.Join(lines, "\n")
}

func uniqueAppend(a, b []int) []int {
	set := map[int]bool{}
	for _, v := range a {
		set[v] = true
	}
	for _, v := range b {
		set[v] = true
	}
	var result []int
	for v := range set {
		result = append(result, v)
	}
	sort.Ints(result)
	return result
}

func contains(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func remove(s []int, v int) []int {
	var result []int
	for _, x := range s {
		if x != v {
			result = append(result, x)
		}
	}
	return result
}

var tasks *TaskManager

// ========== 工具 ==========

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

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func runBash(command string) string {
	for _, d := range []string{"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"} {
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

func dispatchTool(name string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	num := func(k string) int { v, _ := p[k].(float64); return int(v) }
	intList := func(k string) []int {
		arr, _ := p[k].([]any)
		var result []int
		for _, v := range arr {
			if f, ok := v.(float64); ok {
				result = append(result, int(f))
			}
		}
		return result
	}
	switch name {
	case "bash":
		return runBash(str("command"))
	case "read_file":
		return runRead(str("path"), num("limit"))
	case "write_file":
		return runWrite(str("path"), str("content"))
	case "edit_file":
		return runEdit(str("path"), str("old_text"), str("new_text"))
	case "task_create":
		return tasks.Create(str("subject"), str("description"))
	case "task_update":
		return tasks.Update(num("task_id"), str("status"), intList("addBlockedBy"), intList("addBlocks"))
	case "task_list":
		return tasks.ListAll()
	case "task_get":
		return tasks.Get(num("task_id"))
	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

var toolDefs = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{Name: "bash", Description: anthropic.String("Run a shell command."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"command": map[string]any{"type": "string"}}, Required: []string{"command"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_file", Description: anthropic.String("Read file contents."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, Required: []string{"path"}}}},
	{OfTool: &anthropic.ToolParam{Name: "write_file", Description: anthropic.String("Write content to file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}}, Required: []string{"path", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "edit_file", Description: anthropic.String("Replace exact text in file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "old_text": map[string]any{"type": "string"}, "new_text": map[string]any{"type": "string"}}, Required: []string{"path", "old_text", "new_text"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_create", Description: anthropic.String("Create a new task."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"subject": map[string]any{"type": "string"}, "description": map[string]any{"type": "string"}}, Required: []string{"subject"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_update", Description: anthropic.String("Update a task's status or dependencies."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}, "status": map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed"}}, "addBlockedBy": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}}, "addBlocks": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}}}, Required: []string{"task_id"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_list", Description: anthropic.String("List all tasks with status summary."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_get", Description: anthropic.String("Get full details of a task by ID."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}}, Required: []string{"task_id"}}}},
}

func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	for {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model: anthropic.Model(modelID), System: []anthropic.TextBlockParam{{Text: systemMsg}},
			Messages: *messages, Tools: toolDefs, MaxTokens: 8000,
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
	tasks = NewTaskManager(tasksDir)
	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[36ms07-go >> \033[0m")
		if !scanner.Scan() {
			break
		}
		query := strings.TrimSpace(scanner.Text())
		if query == "" || query == "q" || query == "exit" {
			break
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
