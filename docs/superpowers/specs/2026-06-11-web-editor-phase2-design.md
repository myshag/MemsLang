# Web Viewer — Phase 2: in-browser SOIDL editor + compile-on-demand

**Date:** 2026-06-11
**Status:** Approved by user ("да делай") — design agreed in conversation
**Builds on:** phase-1 viewer (spec 2026-06-11-web-viewer-design.md)

## Goal
Edit SOIDL source in the browser and recompile on demand: editor panel with
line numbers, a Compile button, compile/FEM errors shown under the editor,
result rendered immediately in the existing 3D viewer.

## Scope
- **Exporter:** `build_bundle_from_source(src, n_modes=3)` — same bundle from
  a source string (refactor of the path-based `_compile`).
- **Backend:**
  - `GET /api/source/{name}` → `{name, source}` (raw .soidl text of an example).
  - `POST /api/compile` `{source}` → bundle JSON; on failure HTTP 422 with
    `{detail}` (parse/elaboration/FEM error text, including soidlc's
    error report lines).
  - Bundle cache for examples keyed by `(name, file mtime)` so editing
    `examples/*.soidl` on disk is picked up without a backend restart.
- **Frontend:** collapsible editor panel (CodeMirror 6 via
  `@uiw/react-codemirror`, dark theme, line numbers; no SOIDL grammar yet —
  YAGNI), loads the selected example's source, **Compile** button
  (and Cmd/Ctrl+Enter), spinner while compiling, error box under the editor,
  successful compile replaces the current bundle in the viewer.

## Non-goals
- SOIDL syntax highlighting/grammar, autocomplete (later).
- Click-error-to-line navigation (errors shown as text).
- Saving edited source back to disk; multi-file projects; auth.

## Error handling
Compile failures must surface the compiler's own report (errors list /
exception text) verbatim in the UI — never a bare 500/blank.

## Testing
- Exporter: `build_bundle_from_source` returns a valid bundle for an example
  read from disk; broken source raises.
- Backend: source endpoint 200/404; compile endpoint 200 with valid bundle on
  good source, 422 with detail on broken source; mtime cache: bundle re-built
  after example file is touched with modified content (tmp example).
- Frontend: build passes; e2e screenshot of editor + successful recompile.
