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
  const data = {
    path: doc.uri.fsPath,
    language: doc.languageId,
    content: doc.getText().slice(0, max),
    selection: doc.getText(ed.selection).slice(0, max),
    line: ed.selection.active.line + 1,
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
