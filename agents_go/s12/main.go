// s12 - Worktree + Task Isolation (Go version)
// 控制面（任务）+ 执行面（worktree）分离，通过 task_id 绑定。
// 运行: cd agents_go && go run ./s12/

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
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
	repoRoot string
	modelID  string
	client   anthropic.Client
	systemMsg string
)

func init() {
	workdir, _ = os.Getwd()
	modelID = os.Getenv("MODEL_ID")
	if modelID == "" {
		modelID = "claude-sonnet-4-6"
	}
	repoRoot = detectRepoRoot(workdir)
	if repoRoot == "" {
		repoRoot = workdir
	}
	systemMsg = fmt.Sprintf(
		"You are a coding agent at %s. "+
			"Use task + worktree tools for multi-task work. "+
			"For parallel or risky changes: create tasks, allocate worktree lanes, "+
			"run commands in those lanes, then choose keep/remove for closeout. "+
			"Use worktree_events when you need lifecycle visibility.", workdir)
	opts := []option.RequestOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client = anthropic.NewClient(opts...)
}

func detectRepoRoot(cwd string) string {
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	root := strings.TrimSpace(string(out))
	if info, err := os.Stat(root); err == nil && info.IsDir() {
		return root
	}
	return ""
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// ========== EventBus ==========

type EventBus struct {
	path string
}

func NewEventBus(path string) *EventBus {
	os.MkdirAll(filepath.Dir(path), 0755)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		os.WriteFile(path, []byte(""), 0644)
	}
	return &EventBus{path: path}
}

func (eb *EventBus) Emit(event string, task, worktree map[string]any, errMsg string) {
	payload := map[string]any{
		"event": event, "ts": time.Now().Unix(),
		"task": task, "worktree": worktree,
	}
	if task == nil {
		payload["task"] = map[string]any{}
	}
	if worktree == nil {
		payload["worktree"] = map[string]any{}
	}
	if errMsg != "" {
		payload["error"] = errMsg
	}
	data, _ := json.Marshal(payload)
	f, _ := os.OpenFile(eb.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	defer f.Close()
	f.Write(data)
	f.WriteString("\n")
}

func (eb *EventBus) ListRecent(limit int) string {
	if limit <= 0 {
		limit = 20
	}
	if limit > 200 {
		limit = 200
	}
	data, _ := os.ReadFile(eb.path)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) > limit {
		lines = lines[len(lines)-limit:]
	}
	var items []any
	for _, line := range lines {
		if line == "" {
			continue
		}
		var obj any
		if json.Unmarshal([]byte(line), &obj) == nil {
			items = append(items, obj)
		}
	}
	out, _ := json.MarshalIndent(items, "", "  ")
	return string(out)
}

// ========== TaskManager ==========

type Task struct {
	ID          int     `json:"id"`
	Subject     string  `json:"subject"`
	Description string  `json:"description"`
	Status      string  `json:"status"`
	Owner       string  `json:"owner"`
	Worktree    string  `json:"worktree"`
	BlockedBy   []int   `json:"blockedBy"`
	CreatedAt   float64 `json:"created_at"`
	UpdatedAt   float64 `json:"updated_at"`
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
	data, err := os.ReadFile(filepath.Join(tm.dir, fmt.Sprintf("task_%d.json", taskID)))
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

func (tm *TaskManager) Exists(taskID int) bool {
	_, err := os.Stat(filepath.Join(tm.dir, fmt.Sprintf("task_%d.json", taskID)))
	return err == nil
}

func (tm *TaskManager) Create(subject, description string) string {
	now := float64(time.Now().Unix())
	t := &Task{ID: tm.nextID, Subject: subject, Description: description, Status: "pending", CreatedAt: now, UpdatedAt: now}
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

func (tm *TaskManager) Update(taskID int, status, owner string) string {
	t, err := tm.load(taskID)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	if status != "" {
		if status != "pending" && status != "in_progress" && status != "completed" {
			return fmt.Sprintf("Error: invalid status '%s'", status)
		}
		t.Status = status
	}
	if owner != "" {
		t.Owner = owner
	}
	t.UpdatedAt = float64(time.Now().Unix())
	tm.save(t)
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
}

func (tm *TaskManager) BindWorktree(taskID int, worktree, owner string) string {
	t, err := tm.load(taskID)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	t.Worktree = worktree
	if owner != "" {
		t.Owner = owner
	}
	if t.Status == "pending" {
		t.Status = "in_progress"
	}
	t.UpdatedAt = float64(time.Now().Unix())
	tm.save(t)
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
}

func (tm *TaskManager) UnbindWorktree(taskID int) string {
	t, err := tm.load(taskID)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	t.Worktree = ""
	t.UpdatedAt = float64(time.Now().Unix())
	tm.save(t)
	data, _ := json.MarshalIndent(t, "", "  ")
	return string(data)
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
		owner := ""
		if t.Owner != "" {
			owner = " owner=" + t.Owner
		}
		wt := ""
		if t.Worktree != "" {
			wt = " wt=" + t.Worktree
		}
		lines = append(lines, fmt.Sprintf("%s #%d: %s%s%s", m, t.ID, t.Subject, owner, wt))
	}
	return strings.Join(lines, "\n")
}

// ========== WorktreeManager ==========

type WtEntry struct {
	Name      string  `json:"name"`
	Path      string  `json:"path"`
	Branch    string  `json:"branch"`
	TaskID    *int    `json:"task_id"`
	Status    string  `json:"status"`
	CreatedAt float64 `json:"created_at,omitempty"`
	RemovedAt float64 `json:"removed_at,omitempty"`
	KeptAt    float64 `json:"kept_at,omitempty"`
}

type WtIndex struct {
	Worktrees []WtEntry `json:"worktrees"`
}

type WorktreeManager struct {
	repoRoot     string
	dir          string
	indexPath    string
	tasks        *TaskManager
	events       *EventBus
	gitAvailable bool
}

func NewWorktreeManager(repoRoot string, tasks *TaskManager, events *EventBus) *WorktreeManager {
	dir := filepath.Join(repoRoot, ".worktrees")
	os.MkdirAll(dir, 0755)
	indexPath := filepath.Join(dir, "index.json")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		os.WriteFile(indexPath, []byte(`{"worktrees":[]}`), 0644)
	}
	wm := &WorktreeManager{repoRoot: repoRoot, dir: dir, indexPath: indexPath, tasks: tasks, events: events}
	wm.gitAvailable = wm.isGitRepo()
	return wm
}

func (wm *WorktreeManager) isGitRepo() bool {
	cmd := exec.Command("git", "rev-parse", "--is-inside-work-tree")
	cmd.Dir = wm.repoRoot
	err := cmd.Run()
	return err == nil
}

func (wm *WorktreeManager) runGit(args []string) (string, error) {
	if !wm.gitAvailable {
		return "", fmt.Errorf("not in a git repository")
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = wm.repoRoot
	out, err := cmd.CombinedOutput()
	result := strings.TrimSpace(string(out))
	if err != nil {
		if result != "" {
			return "", fmt.Errorf("%s", result)
		}
		return "", fmt.Errorf("git %s failed: %v", strings.Join(args, " "), err)
	}
	if result == "" {
		return "(no output)", nil
	}
	return result, nil
}

func (wm *WorktreeManager) loadIndex() WtIndex {
	data, _ := os.ReadFile(wm.indexPath)
	var idx WtIndex
	json.Unmarshal(data, &idx)
	return idx
}

func (wm *WorktreeManager) saveIndex(idx WtIndex) {
	data, _ := json.MarshalIndent(idx, "", "  ")
	os.WriteFile(wm.indexPath, data, 0644)
}

func (wm *WorktreeManager) find(name string) *WtEntry {
	idx := wm.loadIndex()
	for i := range idx.Worktrees {
		if idx.Worktrees[i].Name == name {
			return &idx.Worktrees[i]
		}
	}
	return nil
}

var validName = regexp.MustCompile(`^[A-Za-z0-9._-]{1,40}$`)

func (wm *WorktreeManager) Create(name string, taskID int, baseRef string) string {
	if !validName.MatchString(name) {
		return "Error: Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
	}
	if wm.find(name) != nil {
		return fmt.Sprintf("Error: Worktree '%s' already exists", name)
	}
	hasTask := taskID > 0
	if hasTask && !wm.tasks.Exists(taskID) {
		return fmt.Sprintf("Error: Task %d not found", taskID)
	}
	if baseRef == "" {
		baseRef = "HEAD"
	}

	path := filepath.Join(wm.dir, name)
	branch := "wt/" + name

	taskInfo := map[string]any{}
	if hasTask {
		taskInfo["id"] = taskID
	}
	wm.events.Emit("worktree.create.before", taskInfo, map[string]any{"name": name, "base_ref": baseRef}, "")

	_, err := wm.runGit([]string{"worktree", "add", "-b", branch, path, baseRef})
	if err != nil {
		wm.events.Emit("worktree.create.failed", taskInfo, map[string]any{"name": name}, err.Error())
		return fmt.Sprintf("Error: %v", err)
	}

	entry := WtEntry{Name: name, Path: path, Branch: branch, Status: "active", CreatedAt: float64(time.Now().Unix())}
	if hasTask {
		entry.TaskID = &taskID
	}

	idx := wm.loadIndex()
	idx.Worktrees = append(idx.Worktrees, entry)
	wm.saveIndex(idx)

	if hasTask {
		wm.tasks.BindWorktree(taskID, name, "")
	}

	wm.events.Emit("worktree.create.after", taskInfo,
		map[string]any{"name": name, "path": path, "branch": branch, "status": "active"}, "")

	data, _ := json.MarshalIndent(entry, "", "  ")
	return string(data)
}

func (wm *WorktreeManager) ListAll() string {
	idx := wm.loadIndex()
	if len(idx.Worktrees) == 0 {
		return "No worktrees in index."
	}
	var lines []string
	for _, wt := range idx.Worktrees {
		suffix := ""
		if wt.TaskID != nil {
			suffix = fmt.Sprintf(" task=%d", *wt.TaskID)
		}
		lines = append(lines, fmt.Sprintf("[%s] %s -> %s (%s)%s", wt.Status, wt.Name, wt.Path, wt.Branch, suffix))
	}
	return strings.Join(lines, "\n")
}

func (wm *WorktreeManager) Status(name string) string {
	wt := wm.find(name)
	if wt == nil {
		return fmt.Sprintf("Error: Unknown worktree '%s'", name)
	}
	if _, err := os.Stat(wt.Path); os.IsNotExist(err) {
		return fmt.Sprintf("Error: Worktree path missing: %s", wt.Path)
	}
	cmd := exec.Command("git", "status", "--short", "--branch")
	cmd.Dir = wt.Path
	out, _ := cmd.CombinedOutput()
	result := strings.TrimSpace(string(out))
	if result == "" {
		return "Clean worktree"
	}
	return result
}

func (wm *WorktreeManager) Run(name, command string) string {
	for _, d := range []string{"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"} {
		if strings.Contains(command, d) {
			return "Error: Dangerous command blocked"
		}
	}
	wt := wm.find(name)
	if wt == nil {
		return fmt.Sprintf("Error: Unknown worktree '%s'", name)
	}
	if _, err := os.Stat(wt.Path); os.IsNotExist(err) {
		return fmt.Sprintf("Error: Worktree path missing: %s", wt.Path)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
	cmd.Dir = wt.Path
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "Error: Timeout (300s)"
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

func (wm *WorktreeManager) Remove(name string, force, completeTask bool) string {
	wt := wm.find(name)
	if wt == nil {
		return fmt.Sprintf("Error: Unknown worktree '%s'", name)
	}

	taskInfo := map[string]any{}
	if wt.TaskID != nil {
		taskInfo["id"] = *wt.TaskID
	}
	wm.events.Emit("worktree.remove.before", taskInfo, map[string]any{"name": name, "path": wt.Path}, "")

	args := []string{"worktree", "remove"}
	if force {
		args = append(args, "--force")
	}
	args = append(args, wt.Path)
	_, err := wm.runGit(args)
	if err != nil {
		wm.events.Emit("worktree.remove.failed", taskInfo, map[string]any{"name": name}, err.Error())
		return fmt.Sprintf("Error: %v", err)
	}

	if completeTask && wt.TaskID != nil {
		tid := *wt.TaskID
		wm.tasks.Update(tid, "completed", "")
		wm.tasks.UnbindWorktree(tid)
		t, _ := wm.tasks.load(tid)
		subject := ""
		if t != nil {
			subject = t.Subject
		}
		wm.events.Emit("task.completed",
			map[string]any{"id": tid, "subject": subject, "status": "completed"},
			map[string]any{"name": name}, "")
	}

	idx := wm.loadIndex()
	for i := range idx.Worktrees {
		if idx.Worktrees[i].Name == name {
			idx.Worktrees[i].Status = "removed"
			idx.Worktrees[i].RemovedAt = float64(time.Now().Unix())
		}
	}
	wm.saveIndex(idx)

	wm.events.Emit("worktree.remove.after", taskInfo,
		map[string]any{"name": name, "path": wt.Path, "status": "removed"}, "")
	return fmt.Sprintf("Removed worktree '%s'", name)
}

func (wm *WorktreeManager) Keep(name string) string {
	wt := wm.find(name)
	if wt == nil {
		return fmt.Sprintf("Error: Unknown worktree '%s'", name)
	}
	idx := wm.loadIndex()
	var kept *WtEntry
	for i := range idx.Worktrees {
		if idx.Worktrees[i].Name == name {
			idx.Worktrees[i].Status = "kept"
			idx.Worktrees[i].KeptAt = float64(time.Now().Unix())
			kept = &idx.Worktrees[i]
		}
	}
	wm.saveIndex(idx)

	taskInfo := map[string]any{}
	if wt.TaskID != nil {
		taskInfo["id"] = *wt.TaskID
	}
	wm.events.Emit("worktree.keep", taskInfo,
		map[string]any{"name": name, "path": wt.Path, "status": "kept"}, "")

	data, _ := json.MarshalIndent(kept, "", "  ")
	return string(data)
}

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

// ========== 全局实例 ==========

var (
	tasks     *TaskManager
	events    *EventBus
	worktrees *WorktreeManager
)

// ========== 工具分发（16 个） ==========

func dispatchTool(name string, input json.RawMessage) string {
	var p map[string]any
	json.Unmarshal(input, &p)
	str := func(k string) string { v, _ := p[k].(string); return v }
	num := func(k string) int { v, _ := p[k].(float64); return int(v) }
	boolVal := func(k string) bool { v, _ := p[k].(bool); return v }

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
	case "task_list":
		return tasks.ListAll()
	case "task_get":
		return tasks.Get(num("task_id"))
	case "task_update":
		return tasks.Update(num("task_id"), str("status"), str("owner"))
	case "task_bind_worktree":
		return tasks.BindWorktree(num("task_id"), str("worktree"), str("owner"))
	case "worktree_create":
		return worktrees.Create(str("name"), num("task_id"), str("base_ref"))
	case "worktree_list":
		return worktrees.ListAll()
	case "worktree_status":
		return worktrees.Status(str("name"))
	case "worktree_run":
		return worktrees.Run(str("name"), str("command"))
	case "worktree_keep":
		return worktrees.Keep(str("name"))
	case "worktree_remove":
		return worktrees.Remove(str("name"), boolVal("force"), boolVal("complete_task"))
	case "worktree_events":
		return events.ListRecent(num("limit"))
	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

var toolDefs = []anthropic.ToolUnionParam{
	{OfTool: &anthropic.ToolParam{Name: "bash", Description: anthropic.String("Run a shell command in the current workspace (blocking)."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"command": map[string]any{"type": "string"}}, Required: []string{"command"}}}},
	{OfTool: &anthropic.ToolParam{Name: "read_file", Description: anthropic.String("Read file contents."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, Required: []string{"path"}}}},
	{OfTool: &anthropic.ToolParam{Name: "write_file", Description: anthropic.String("Write content to file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "content": map[string]any{"type": "string"}}, Required: []string{"path", "content"}}}},
	{OfTool: &anthropic.ToolParam{Name: "edit_file", Description: anthropic.String("Replace exact text in file."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"path": map[string]any{"type": "string"}, "old_text": map[string]any{"type": "string"}, "new_text": map[string]any{"type": "string"}}, Required: []string{"path", "old_text", "new_text"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_create", Description: anthropic.String("Create a new task on the shared task board."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"subject": map[string]any{"type": "string"}, "description": map[string]any{"type": "string"}}, Required: []string{"subject"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_list", Description: anthropic.String("List all tasks with status, owner, and worktree binding."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_get", Description: anthropic.String("Get task details by ID."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}}, Required: []string{"task_id"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_update", Description: anthropic.String("Update task status or owner."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}, "status": map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed"}}, "owner": map[string]any{"type": "string"}}, Required: []string{"task_id"}}}},
	{OfTool: &anthropic.ToolParam{Name: "task_bind_worktree", Description: anthropic.String("Bind a task to a worktree name."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"task_id": map[string]any{"type": "integer"}, "worktree": map[string]any{"type": "string"}, "owner": map[string]any{"type": "string"}}, Required: []string{"task_id", "worktree"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_create", Description: anthropic.String("Create a git worktree and optionally bind it to a task."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}, "task_id": map[string]any{"type": "integer"}, "base_ref": map[string]any{"type": "string"}}, Required: []string{"name"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_list", Description: anthropic.String("List worktrees tracked in index."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_status", Description: anthropic.String("Show git status for one worktree."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}}, Required: []string{"name"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_run", Description: anthropic.String("Run a shell command in a named worktree directory."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}, "command": map[string]any{"type": "string"}}, Required: []string{"name", "command"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_remove", Description: anthropic.String("Remove a worktree and optionally mark its bound task completed."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}, "force": map[string]any{"type": "boolean"}, "complete_task": map[string]any{"type": "boolean"}}, Required: []string{"name"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_keep", Description: anthropic.String("Mark a worktree as kept without removing it."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"name": map[string]any{"type": "string"}}, Required: []string{"name"}}}},
	{OfTool: &anthropic.ToolParam{Name: "worktree_events", Description: anthropic.String("List recent worktree/task lifecycle events."),
		InputSchema: anthropic.ToolInputSchemaParam{Properties: map[string]any{"limit": map[string]any{"type": "integer"}}}}},
}

// ========== Agent Loop ==========

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

	tasks = NewTaskManager(filepath.Join(repoRoot, ".tasks"))
	events = NewEventBus(filepath.Join(repoRoot, ".worktrees", "events.jsonl"))
	worktrees = NewWorktreeManager(repoRoot, tasks, events)

	fmt.Printf("Repo root for s12: %s\n", repoRoot)
	if !worktrees.gitAvailable {
		fmt.Println("Note: Not in a git repo. worktree_* tools will return errors.")
	}

	var history []anthropic.MessageParam
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[36ms12-go >> \033[0m")
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
