/**
 * Entry point. `?sandbox` runs the real app around a scripted fake SOMA with an in-memory fact log (like
 * tools/sandbox.py's temp db); otherwise SOMA (tfjs + OCR) loads lazily and the fact log lives in IndexedDB.
 */
import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/inter";
import "./index.css";
import App from "./App";
import { FakeSoma } from "./pette/fakeSoma";
import { loadSoma } from "./pette/loadSoma";
import { FactStore } from "./pette/world/store";

const root = ReactDOM.createRoot(document.getElementById("root")!);
const sandbox = new URLSearchParams(window.location.search).has("sandbox");

function Message({ title, text }: { title: string; text: string }) {
  return (
    <main className="flex h-full items-center justify-center bg-zinc-950 p-6 text-center">
      <div className="max-w-sm space-y-1">
        <div className="text-sm font-semibold text-zinc-50">{title}</div>
        <div className="text-xs text-zinc-400">{text}</div>
      </div>
    </main>
  );
}

async function boot() {
  root.render(<Message title="Pette" text={sandbox ? "Starting the sandbox…" : "Loading SOMA…"} />);
  try {
    const store = sandbox ? new FactStore() : await FactStore.openOrMemory();
    const soma = sandbox ? new FakeSoma() : await loadSoma();
    // ?debug exposes the service for inspection from DevTools (status, suggestion); nothing else changes.
    if (new URLSearchParams(window.location.search).has("debug")) (window as unknown as { __soma: unknown }).__soma = soma;
    root.render(
      <React.StrictMode>
        <App soma={soma} store={store} sandbox={sandbox} />
      </React.StrictMode>,
    );
  } catch (e) {
    console.error(e);
    root.render(<Message title="SOMA couldn't start" text="Reload the page to try again. Pette needs a recent Chrome, Edge or Firefox." />);
  }
}

void boot();
