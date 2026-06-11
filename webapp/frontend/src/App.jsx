import React, { useEffect, useState } from 'react'
import DeviceCanvas from './DeviceCanvas.jsx'
import ResultPanel from './ResultPanel.jsx'
import Controls from './Controls.jsx'
import ExampleSelector from './ExampleSelector.jsx'

const DEFAULTS = { scale: 40, speed: 1.5, wireframe: false, colorField: 'disp_mag' }

export default function App() {
  const [examples, setExamples] = useState([])
  const [name, setName] = useState('')
  const [bundle, setBundle] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(-1)
  const [settings, setSettings] = useState(DEFAULTS)

  useEffect(() => { fetch('/api/examples').then(r => r.json()).then(setExamples) }, [])
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
      {loading && <span>compiling…</span>}
      {error && <span style={{color:'#f66'}}>{error}</span>}
    </div>
    <div style={{display:'grid', gridTemplateColumns:'240px 1fr', minHeight:0}}>
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
