// s01 - The Agent Loop (Go version)
// 最小 Agent：一个 while 循环 + 一个 bash 工具。
// 运行: cd agents_go && go run ./s01/

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
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
	systemMsg = fmt.Sprintf("You are a coding agent at %s. Use bash to solve tasks. Act, don't explain.", workdir)

	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

// 工具定义：只有一个 bash
var tools = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{
		Name:        "bash",
		Description: anthropic.String("Run a shell command."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{"command": map[string]any{"type": "string"}},
			Required:   []string{"command"},
		},
	}},
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
	if len(result) > 50000 {
		return result[:50000]
	}
	return result
}

// 核心：agent loop
func agentLoop(messages *[]anthropic.MessageParam) error {
	ctx := context.Background()
	for {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(modelID),
			System:    []anthropic.TextBlockParam{{Text: systemMsg}},
			Messages:  *messages,
			Tools:     tools,
			MaxTokens: 8000,
		})
		if err != nil {
			return fmt.Errorf("API error: %w", err)
		}

		// 追加 assistant 回复
		var blocks []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			switch b.Type {
			case "text":
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{Text: b.AsText().Text},
				})
			case "tool_use":
				tu := b.AsToolUse()
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{ID: tu.ID, Name: tu.Name, Input: tu.Input},
				})
			}
		}
		*messages = append(*messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleAssistant, Content: blocks})

		if resp.StopReason != anthropic.StopReasonToolUse {
			return nil
		}

		// 执行工具
		var results []anthropic.ContentBlockParamUnion
		for _, b := range resp.Content {
			if b.Type == "tool_use" {
				tu := b.AsToolUse()
				var params map[string]any
				json.Unmarshal(tu.Input, &params)
				command, _ := params["command"].(string)
				fmt.Printf("\033[33m$ %s\033[0m\n", command)
				output := runBash(command)
				if len(output) > 200 {
					fmt.Println(output[:200])
				} else {
					fmt.Println(output)
				}
				results = append(results, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: tu.ID,
						Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: output}}},
					},
				})
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
	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[36ms01-go >> \033[0m")
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
