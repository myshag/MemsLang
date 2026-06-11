import React, { useEffect, useState, useCallback } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { EditorView } from '@codemirror/view'

const CM_EXTENSIONS = [EditorView.lineWrapping]

export default function EditorPanel({ name, onBundle }) {
  const [source, setSource] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!name) return
    setError(null)
    fetch(`/api/source/${name}`).then(r => r.json()).then(d => setSource(d.source))
  }, [name])

  const compile = useCallback(() => {
    setBusy(true); setError(null)
    fetch('/api/compile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    })
      .then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e.detail || `HTTP ${r.status}`)); return r.json() })
      .then(onBundle)
      .catch(e => setError(String(e)))
      .finally(() => setBusy(false))
  }, [source, onBundle])

  const onKey = useCallback((e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); compile() }
  }, [compile])

  return <div style={{display:'flex', flexDirection:'column', minHeight:0, height:'100%'}}
              onKeyDown={onKey}>
    <div style={{display:'flex', gap:8, alignItems:'center', padding:'4px 0'}}>
      <button onClick={compile} disabled={busy || !source}
        style={{padding:'4px 14px', fontWeight:'bold'}}>
        {busy ? 'compiling…' : 'Compile ⌘⏎'}
      </button>
    </div>
    <div style={{flex:1, minHeight:0, overflow:'auto', border:'1px solid #333',
                 fontSize:13}}>
      <CodeMirror value={source} onChange={setSource} theme="dark"
        height="100%" extensions={CM_EXTENSIONS}
        basicSetup={{ lineNumbers: true, foldGutter: false }}/>
    </div>
    {error && <pre style={{color:'#f66', whiteSpace:'pre-wrap', maxHeight:120,
      overflow:'auto', margin:'6px 0 0', fontSize:12}}>{error}</pre>}
  </div>
}
