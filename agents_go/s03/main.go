// s03 - TodoWrite (Go version)
// TodoManager + nag reminder，让 Agent 不偏航。
// 运行: cd agents_go && go run ./s03/

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
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

var (
	workdir   string
	modelID   string
	client    anthropic.Client
	systemMsg string
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	systemMsg = fmt.Sprintf(`You are a coding agent at %s.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose.`, workdir)
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

// ========== TodoManager ==========

type TodoItem struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Status string `json:"status"`
}

type TodoManager struct {
	items []TodoItem
}

func (tm *TodoManager) Update(items []any) (string, error) {
	if len(items) > 20 {
		return "", fmt.Errorf("max 20 todos allowed")
	}
	var validated []TodoItem
	inProgressCount := 0
	for i, raw := range items {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		text, _ := m["text"].(string)
		status, _ := m["status"].(string)
		id, _ := m["id"].(string)
		if text == "" {
			return "", fmt.Errorf("item %d: text required", i+1)
		}
		if status == "" {
			status = "pending"
		}
		if status != "pending" && status != "in_progress" && status != "completed" {
			return "", fmt.Errorf("item %s: invalid status '%s'", id, status)
		}
		if status == "in_progress" {
			inProgressCount++
		}
		if id == "" {
			id = fmt.Sprintf("%d", i+1)
		}
		validated = append(validated, TodoItem{ID: id, Text: text, Status: status})
	}
	if inProgressCount > 1 {
		return "", fmt.Errorf("only one task can be in_progress at a time")
	}
	tm.items = validated
	return tm.Render(), nil
}

func (tm *TodoManager) Render() string {
	if len(tm.items) == 0 {
		return "No todos."
	}
	markers := map[string]string{"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
	var lines []string
	done := 0
	for _, item := range tm.items {
		m := markers[item.Status]
		if m == "" {
			m = "[?]"
		}
		lines = append(lines, fmt.Sprintf("%s #%s: %s", m, item.ID, item.Text))
		if item.Status == "completed" {
			done++
		}
	}
	lines = append(lines, fmt.Sprintf("\n(%d/%d completed)", done, len(tm.items)))
	return strings.Join(lines, "\n")
}

var todo = &TodoManager{}

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

	switch name {
	case "bash":
		return runBash(str("command"))
	case "read_file":
		return runRead(str("path"), num("limit"))
	case "write_file":
		return runWrite(str("path"), str("content"))
	case "edit_file":
		return runEdit(str("path"), str("old_text"), str("new_text"))
	case "todo":
		items, _ := p["items"].([]any)
		result, err := todo.Update(items)
		if err != nil {
			return fmt.Sprintf("Error: %v", err)
		}
		return result
	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

var tools = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{Name: "bash", Description: anthropic.String("Run a shell command."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"command": map[string]any{"type": "string"}}, Required: []string{"command"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_file", Description: anthropic.String("Read file contents."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, Required: []string{"path"}}}},
	{OfTool: &anthropic.ToolParam{Name: "write_file", Description: anthropic.String("Write content to file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}}, Required: []string{"path", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "edit_file", Description: anthropic.String("Replace exact text in file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "old_text": map[string]any{"type": "string"}, "new_text": map[string]any{"type": "string"}}, Required: []string{"path", "old_text", "new_text"}}}},
	{OfTool: &anthropic.ToolParam{Name: "todo", Description: anthropic.String("Update task list. Track progress on multi-step tasks."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"items": map[string]any{"type": "array", "items": map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string"}, "text": map[string]any{"type": "string"}, "status": map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed"}}}, "required": []string{"id", "text", "status"}}}}, Required: []string{"items"}}}},
}

// ========== Agent Loop + Nag ==========

func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	roundsSinceTodo := 0
	for {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model: anthropic.Model(modelID), System: []anthropic.TextBlockParam{{Text: systemMsg}},
			Messages: *messages, Tools: tools, MaxTokens: 8000,
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
		usedTodo := false
		for _, b := range resp.Content {
			if b.Type == "tool_use" {
				tu := b.AsToolUse()
				output := dispatchTool(tu.Name, tu.Input)
				fmt.Printf("> %s: %s\n", tu.Name, truncate(output, 200))
				results = append(results, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: tu.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}}}})
				if tu.Name == "todo" {
					usedTodo = true
				}
			}
		}
		if usedTodo {
			roundsSinceTodo = 0
		} else {
			roundsSinceTodo++
		}
		// Nag reminder
		if roundsSinceTodo >= 3 {
			results = append([]anthropic.ContentBlockParamUnion{
				{OfText: &anthropic.TextBlockParam{Text: "<reminder>Update your todos.</reminder>"}},
			}, results...)
		}
		*messages = append(*messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: results})
	}
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY not set")
		os.Exit(1)
	}
	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[36ms03-go >> \033[0m")
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
