// s06 - Context Compact (Go version)
// 三层压缩：micro_compact + auto_compact + manual compact 工具。
// 运行: cd agents_go && go run ./s06/

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
	workdir       string
	modelID       string
	client        anthropic.Client
	systemMsg     string
	transcriptDir string
)

const (
	threshold  = 50000
	keepRecent = 3
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	systemMsg = fmt.Sprintf("You are a coding agent at %s. Use tools to solve tasks.", workdir)
	transcriptDir = filepath.Join(workdir, ".transcripts")
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

// ========== Token 估算 ==========

func estimateTokens(messages []anthropic.MessageParam) int {
	data, _ := json.Marshal(messages)
	return len(data) / 4
}

// ========== Layer 1: micro_compact ==========

func microCompact(messages []anthropic.MessageParam) {
	// 收集所有 tool_result 的位置
	type toolResultRef struct {
		msgIdx  int
		partIdx int
	}
	var refs []toolResultRef
	for i, msg := range messages {
		if msg.Role != anthropic.MessageParamRoleUser {
			continue
		}
		for j, block := range msg.Content {
			if block.OfToolResult != nil {
				refs = append(refs, toolResultRef{i, j})
			}
		}
	}
	if len(refs) <= keepRecent {
		return
	}
	toClear := refs[:len(refs)-keepRecent]
	for _, ref := range toClear {
		tr := messages[ref.msgIdx].Content[ref.partIdx].OfToolResult
		if tr != nil && len(tr.Content) > 0 && tr.Content[0].OfText != nil {
			text := tr.Content[0].OfText.Text
			if len(text) > 100 {
				tr.Content[0].OfText.Text = "[Previous: used tool]"
			}
		}
	}
}

// ========== Layer 2: auto_compact ==========

func autoCompact(messages []anthropic.MessageParam) []anthropic.MessageParam {
	// 保存到磁盘
	os.MkdirAll(transcriptDir, 0755)
	transcriptPath := filepath.Join(transcriptDir, fmt.Sprintf("transcript_%d.jsonl", time.Now().Unix()))
	f, _ := os.Create(transcriptPath)
	for _, msg := range messages {
		data, _ := json.Marshal(msg)
		f.Write(data)
		f.WriteString("\n")
	}
	f.Close()
	fmt.Printf("[transcript saved: %s]\n", transcriptPath)

	// 让 LLM 做摘要
	convText, _ := json.Marshal(messages)
	if len(convText) > 80000 {
		convText = convText[:80000]
	}
	ctx := context.Background()
	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model: anthropic.Model(modelID),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(
				"Summarize this conversation for continuity. Include: 1) What was accomplished, 2) Current state, 3) Key decisions made. Be concise but preserve critical details.\n\n" + string(convText),
			)),
		},
		MaxTokens: 2000,
	})
	summary := "(compression failed)"
	if err == nil && len(resp.Content) > 0 && resp.Content[0].Type == "text" {
		summary = resp.Content[0].AsText().Text
	}

	return []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(
			fmt.Sprintf("[Conversation compressed. Transcript: %s]\n\n%s", transcriptPath, summary),
		)),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("Understood. I have the context from the summary. Continuing.")),
	}
}

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
	case "compact":
		return "Manual compression requested."
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
	{OfTool: &anthropic.ToolParam{Name: "compact", Description: anthropic.String("Trigger manual conversation compression."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"focus": map[string]any{"type": "string", "description": "What to preserve in the summary"}}}}},
}

// ========== Agent Loop ==========

func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	for {
		// Layer 1
		microCompact(*messages)
		// Layer 2
		if estimateTokens(*messages) > threshold {
			fmt.Println("[auto_compact triggered]")
			*messages = autoCompact(*messages)
		}
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
		manualCompact := false
		for _, b := range resp.Content {
			if b.Type == "tool_use" {
				tu := b.AsToolUse()
				if tu.Name == "compact" {
					manualCompact = true
				}
				output := dispatchTool(tu.Name, tu.Input)
				fmt.Printf("> %s: %s\n", tu.Name, truncate(output, 200))
				results = append(results, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: tu.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}}}})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: results})
		// Layer 3
		if manualCompact {
			fmt.Println("[manual compact]")
			*messages = autoCompact(*messages)
		}
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
		fmt.Print("\033[36ms06-go >> \033[0m")
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
