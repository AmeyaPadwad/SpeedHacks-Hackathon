// Keeps ~/.flowstate_current.json in sync with the active editor so the desktop agent can read it on hotkey.
const vscode = require("vscode");
const fs = require("fs");
const os = require("os");
const path = require("path");

const OUT = process.env.FLOWSTATE_ACTIVE_FILE || path.join(os.homedir(), ".flowstate_current.json");
const DEBOUNCE_MS = 400;
let timer;

function config() {
  return vscode.workspace.getConfiguration("flowstate");
}

function clear() {
  fs.rm(OUT, { force: true }, () => {});
}

function write() {
  if (!config().get("enabled")) return clear();
  const ed = vscode.window.activeTextEditor;
  if (!ed || ed.document.uri.scheme !== "file") return;
  const max = config().get("maxChars");
  const doc = ed.document;
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  const severities = ["error", "warning", "info", "hint"];
  // Errors and warnings in the active file, capped so a broken file can't flood the agent prompt.
  const diagnostics = vscode.languages
    .getDiagnostics(doc.uri)
    .filter((d) => d.severity <= vscode.DiagnosticSeverity.Warning)
    .slice(0, 20)
    .map((d) => ({
      line: d.range.start.line + 1,
      severity: severities[d.severity],
      message: d.message.slice(0, 500),
      source: d.source || "",
      code: typeof d.code === "object" ? String(d.code.value) : String(d.code ?? ""),
    }));
  const data = {
    path: doc.uri.fsPath,
    workspace: folder ? folder.uri.fsPath : "",
    language: doc.languageId,
    content: doc.getText().slice(0, max),
    selection: doc.getText(ed.selection).slice(0, max),
    line: ed.selection.active.line + 1,
    diagnostics,
    updatedAt: new Date().toISOString(),
  };
  // Write to a temp file then rename, so the agent never reads a half-written snapshot.
  const tmp = OUT + ".tmp";
  fs.writeFile(tmp, JSON.stringify(data), { encoding: "utf8", mode: 0o600 }, (err) => {
    if (!err) fs.rename(tmp, OUT, () => {});
  });
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(write, DEBOUNCE_MS);
}

function activate(context) {
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(schedule),
    vscode.window.onDidChangeTextEditorSelection(schedule),
    vscode.workspace.onDidSaveTextDocument(schedule),
    vscode.languages.onDidChangeDiagnostics(schedule),
    vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration("flowstate") && schedule()),
    vscode.commands.registerCommand("flowstate.toggle", async () => {
      const on = !config().get("enabled");
      await config().update("enabled", on, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(`FlowState sharing ${on ? "on" : "off"}`);
    })
  );
  schedule();
}

function deactivate() {
  clearTimeout(timer);
  clear();
}

module.exports = { activate, deactivate };
