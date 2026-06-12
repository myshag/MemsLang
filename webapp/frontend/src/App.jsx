import React, { useCallback, useEffect, useState } from 'react'
import DeviceCanvas from './DeviceCanvas.jsx'
import ResultPanel from './ResultPanel.jsx'
import Controls from './Controls.jsx'
import ExampleSelector from './ExampleSelector.jsx'
import EditorPanel from './EditorPanel.jsx'

const DEFAULTS = { scale: 40, speed: 1.5, wireframe: false, colorField: 'disp_mag' }

export default function App() {
  const [examples, setExamples] = useState([])
  const [name, setName] = useState('')
  const [bundle, setBundle] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(-1)
  const [settings, setSettings] = useState(DEFAULTS)
  const [showEditor, setShowEditor] = useState(false)
  // editor width in px, user-resizable via the drag divider
  const [editorWidth, setEditorWidth] = useState(
    () => Math.max(420, Math.round(window.innerWidth * 0.38)))

  const handleBundle = useCallback((b) => { setBundle(b); setActive(-1) }, [])

  const startDrag = useCallback((e) => {
    e.preventDefault()
    const x0 = e.clientX, w0 = editorWidth
    const move = (ev) => setEditorWidth(Math.min(
      Math.max(320, w0 + ev.clientX - x0), window.innerWidth - 500))
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }, [editorWidth])

  // retry the examples fetch until the backend is reachable (it may still
  // be starting up), and surface the failure instead of an empty selector
  useEffect(() => {
    let stop = false
    const load = (attempt = 0) =>
      fetch('/api/examples')
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
        .then(list => { if (!stop) { setExamples(list); setError(null) } })
        .catch(() => {
          if (stop) return
          setError('backend unreachable — retrying…')
          setTimeout(() => load(attempt + 1), Math.min(2000 * (attempt + 1), 10000))
        })
    load()
    return () => { stop = true }
  }, [])
  useEffect(() => {
    if (!name) return
    setBundle(null); setActive(-1); setLoading(true); setError(null)
    fetch(`/api/bundle/${name}`)
      .then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e.detail || r.status)); return r.json() })
      .then(setBundle)
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [name])

  const result = bundle && active >= 0 ? bundle.results[active] : null
  return <div style={{display:'grid', gridTemplateRows:'auto 1fr auto', height:'100%', background:'#0b0d12', color:'#ddd'}}>
    <div style={{display:'flex', gap:16, alignItems:'center', padding:8}}>
      <strong>soidlc viewer</strong>
      <ExampleSelector examples={examples} value={name} onChange={setName}/>
      <button onClick={() => setShowEditor(s => !s)}>{showEditor ? 'hide editor' : 'editor'}</button>
      {loading && <span>compiling…</span>}
      {error && <span style={{color:'#f66'}}>{error}</span>}
    </div>
    <div style={{display:'grid', gridTemplateColumns: showEditor ? `${editorWidth}px 6px 240px 1fr` : '240px 1fr', minHeight:0}}>
      {showEditor && <div style={{minHeight:0, padding:'0 0 0 8px'}}>
        <EditorPanel name={name} onBundle={handleBundle}/>
      </div>}
      {showEditor && <div onPointerDown={startDrag}
        style={{cursor:'col-resize', background:'#2a2d36', borderRadius:3,
                margin:'4px 1px', touchAction:'none'}}/>}
      <div style={{padding:8, overflow:'auto'}}>
        {bundle && <ResultPanel results={bundle.results} active={active} onSelect={setActive}/>}
        {bundle && <Legend layers={bundle.meta.layers}/>}
      </div>
      <div style={{minHeight:0}}>
        {bundle && <DeviceCanvas bundle={bundle} result={result} settings={settings}/>}
      </div>
    </div>
    <Controls settings={settings} set={setSettings} result={result}/>
  </div>
}

function Legend({ layers }) {
  return <div style={{marginTop:16, fontSize:12}}>
    {layers.map((l, i) => <div key={i} style={{display:'flex', gap:6, alignItems:'center'}}>
      <span style={{width:12, height:12, display:'inline-block',
        background:`rgb(${l.color.map(c=>Math.round(c*255)).join(',')})`}}/>
      {l.name}</div>)}
  </div>
}
