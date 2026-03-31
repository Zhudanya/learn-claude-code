// s08_background_tasks.go - Background Tasks (Go version, Anthropic Official SDK)
//
// Go 版本的 s08，使用 Anthropic 官方 Go SDK。
// goroutine 替代 Python 的 threading，sync.Mutex 替代 threading.Lock。
//
// 需要设置环境变量：ANTHROPIC_API_KEY
// 可选：MODEL_ID（默认 claude-sonnet-4-6）、ANTHROPIC_BASE_URL
//
// 运行方式：
//   cd agents_go && go run s08_background_tasks.go

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

// ========== 配置 ==========

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
	systemMsg = fmt.Sprintf("You are a coding agent at %s. Use background_run for long-running commands.", workdir)

	// 创建客户端：自动读取 ANTHROPIC_API_KEY 环境变量
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

// ========== 工具定义 ==========

var tools = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{
		Name:        "bash",
		Description: anthropic.String("Run a shell command (blocking)."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{"command": map[string]any{"type": "string"}},
			Required:   []string{"command"},
		},
	}},
	{OfTool: &anthropic.ToolParam{
		Name:        "read_file",
		Description: anthropic.String("Read file contents."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"path":  map[string]any{"type": "string"},
				"limit": map[string]any{"type": "integer"},
			},
			Required: []string{"path"},
		},
	}},
	{OfTool: &anthropic.ToolParam{
		Name:        "write_file",
		Description: anthropic.String("Write content to file."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"path":    map[string]any{"type": "string"},
				"content": map[string]any{"type": "string"},
			},
			Required: []string{"path", "content"},
		},
	}},
	{OfTool: &anthropic.ToolParam{
		Name:        "edit_file",
		Description: anthropic.String("Replace exact text in file."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"path":     map[string]any{"type": "string"},
				"old_text": map[string]any{"type": "string"},
				"new_text": map[string]any{"type": "string"},
			},
			Required: []string{"path", "old_text", "new_text"},
		},
	}},
	{OfTool: &anthropic.ToolParam{
		Name:        "background_run",
		Description: anthropic.String("Run command in background goroutine. Returns task_id immediately."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{"command": map[string]any{"type": "string"}},
			Required:   []string{"command"},
		},
	}},
	{OfTool: &anthropic.ToolParam{
		Name:        "check_background",
		Description: anthropic.String("Check background task status. Omit task_id to list all."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{"task_id": map[string]any{"type": "string"}},
		},
	}},
}

// ========== 路径安全 ==========

func safePath(p string) (string, error) {
	abs, err := filepath.Abs(filepath.Join(workdir, p))
	if err != nil {
		return "", err
	}
	absWorkdir, _ := filepath.Abs(workdir)
	if !strings.HasPrefix(abs, absWorkdir) {
		return "", fmt.Errorf("path escapes workspace: %s", p)
	}
	return abs, nil
}

// ========== 工具实现 ==========

func shellCmd(ctx context.Context, command string) *exec.Cmd {
	if runtime.GOOS == "windows" {
		return exec.CommandContext(ctx, "cmd", "/C", command)
	}
	return exec.CommandContext(ctx, "sh", "-c", command)
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
	cmd := shellCmd(ctx, command)
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
	if err := os.MkdirAll(filepath.Dir(fp), 0755); err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
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
	content := string(data)
	if !strings.Contains(content, oldText) {
		return fmt.Sprintf("Error: Text not found in %s", path)
	}
	if err := os.WriteFile(fp, []byte(strings.Replace(content, oldText, newText, 1)), 0644); err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return fmt.Sprintf("Edited %s", path)
}

// ========== BackgroundManager ==========

type Notification struct {
	TaskID  string
	Status  string
	Command string
	Result  string
}

type BgTask struct {
	Status  string
	Result  string
	Command string
}

type BackgroundManager struct {
	mu            sync.Mutex
	tasks         map[string]*BgTask
	notifications []Notification
}

func NewBackgroundManager() *BackgroundManager {
	return &BackgroundManager{tasks: make(map[string]*BgTask)}
}

func (bg *BackgroundManager) Run(command string) string {
	taskID := uuid.New().String()[:8]
	bg.mu.Lock()
	bg.tasks[taskID] = &BgTask{Status: "running", Command: command}
	bg.mu.Unlock()

	go bg.execute(taskID, command)
	return fmt.Sprintf("Background task %s started: %s", taskID, truncate(command, 80))
}

func (bg *BackgroundManager) execute(taskID, command string) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	cmd := shellCmd(ctx, command)
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput() // 慢操作

	status := "completed"
	output := strings.TrimSpace(string(out))
	if ctx.Err() == context.DeadlineExceeded {
		output, status = "Error: Timeout (300s)", "timeout"
	} else if err != nil && output == "" {
		output, status = fmt.Sprintf("Error: %v", err), "error"
	}
	if output == "" {
		output = "(no output)"
	}

	bg.mu.Lock()
	bg.tasks[taskID].Status = status
	bg.tasks[taskID].Result = truncate(output, 50000)
	bg.notifications = append(bg.notifications, Notification{
		TaskID: taskID, Status: status,
		Command: truncate(command, 80), Result: truncate(output, 500),
	})
	bg.mu.Unlock()
}

func (bg *BackgroundManager) Check(taskID string) string {
	bg.mu.Lock()
	defer bg.mu.Unlock()
	if taskID != "" {
		t, ok := bg.tasks[taskID]
		if !ok {
			return fmt.Sprintf("Error: Unknown task %s", taskID)
		}
		result := t.Result
		if result == "" {
			result = "(running)"
		}
		return fmt.Sprintf("[%s] %s\n%s", t.Status, truncate(t.Command, 60), result)
	}
	if len(bg.tasks) == 0 {
		return "No background tasks."
	}
	var lines []string
	for tid, t := range bg.tasks {
		lines = append(lines, fmt.Sprintf("%s: [%s] %s", tid, t.Status, truncate(t.Command, 60)))
	}
	return strings.Join(lines, "\n")
}

func (bg *BackgroundManager) DrainNotifications() []Notification {
	bg.mu.Lock()
	defer bg.mu.Unlock()
	notifs := make([]Notification, len(bg.notifications))
	copy(notifs, bg.notifications)
	bg.notifications = bg.notifications[:0]
	return notifs
}

// ========== 工具分发 ==========

var bgManager = NewBackgroundManager()

func dispatchTool(name string, input json.RawMessage) string {
	var params map[string]any
	json.Unmarshal(input, &params)
	str := func(key string) string { v, _ := params[key].(string); return v }
	num := func(key string) int { v, _ := params[key].(float64); return int(v) }

	switch name {
	case "bash":
		return runBash(str("command"))
	case "read_file":
		return runRead(str("path"), num("limit"))
	case "write_file":
		return runWrite(str("path"), str("content"))
	case "edit_file":
		return runEdit(str("path"), str("old_text"), str("new_text"))
	case "background_run":
		return bgManager.Run(str("command"))
	case "check_background":
		return bgManager.Check(str("task_id"))
	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

// ========== Agent Loop ==========

func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	for {
		// 排空后台通知，注入 messages
		notifs := bgManager.DrainNotifications()
		if len(notifs) > 0 && len(*messages) > 0 {
			var parts []string
			for _, n := range notifs {
				parts = append(parts, fmt.Sprintf("[bg:%s] %s: %s", n.TaskID, n.Status, n.Result))
			}
			notifText := strings.Join(parts, "\n")
			*messages = append(*messages,
				anthropic.NewUserMessage(anthropic.NewTextBlock(
					fmt.Sprintf("<background-results>\n%s\n</background-results>", notifText),
				)),
				anthropic.NewAssistantMessage(anthropic.NewTextBlock("Noted background results.")),
			)
		}

		// 调 LLM
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(modelID),
			System:    []anthropic.TextBlockParam{{Text: systemMsg}},
			Messages:  *messages,
			Tools:     tools,
			MaxTokens: 8000,
		})
		if err != nil {
			return fmt.Errorf("API call failed: %w", err)
		}

		// 追加 assistant 回复（保留完整 content 用于后续 messages）
		var assistantBlocks []anthropic.ContentBlockParamUnion
		for _, block := range resp.Content {
			switch block.Type {
			case "text":
				assistantBlocks = append(assistantBlocks,
					anthropic.ContentBlockParamUnion{
						OfText: &anthropic.TextBlockParam{Text: block.AsText().Text},
					})
			case "tool_use":
				tu := block.AsToolUse()
				assistantBlocks = append(assistantBlocks,
					anthropic.ContentBlockParamUnion{
						OfToolUse: &anthropic.ToolUseBlockParam{
							ID: tu.ID, Name: tu.Name, Input: tu.Input,
						},
					})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleAssistant,
			Content: assistantBlocks,
		})

		// 检查是否结束
		if resp.StopReason != anthropic.StopReasonToolUse {
			return nil
		}

		// 执行工具，收集结果
		var toolResults []anthropic.ContentBlockParamUnion
		for _, block := range resp.Content {
			if block.Type == "tool_use" {
				tu := block.AsToolUse()
				output := dispatchTool(tu.Name, tu.Input)
				fmt.Printf("> %s: %s\n", tu.Name, truncate(output, 200))
				toolResults = append(toolResults, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: tu.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{OfText: &anthropic.TextBlockParam{Text: output}},
						},
					},
				})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleUser,
			Content: toolResults,
		})
	}
}

// ========== 主程序 ==========

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY not set")
		os.Exit(1)
	}

	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\033[36ms08-go >> \033[0m")
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

		// 打印最后一条 assistant 消息的文本
		if len(history) > 0 {
			last := history[len(history)-1]
			for _, block := range last.Content {
				if block.OfText != nil {
					fmt.Println(block.OfText.Text)
				}
			}
		}
		fmt.Println()
	}
}
