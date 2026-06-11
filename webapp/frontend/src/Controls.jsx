import React from 'react'
export default function Controls({ settings, set, result }) {
  const f = (k) => (e) => set({ ...settings, [k]: parseFloat(e.target.value) })
  const b = (k) => () => set({ ...settings, [k]: !settings[k] })
  return <div style={{display:'flex', gap:16, alignItems:'center',
                      flexWrap:'wrap', padding:8, color:'#ddd'}}>
    <label>scale {settings.scale}
      <input type="range" min="0" max="200" value={settings.scale} onChange={f('scale')}/></label>
    <label>speed {settings.speed}
      <input type="range" min="0" max="5" step="0.1" value={settings.speed} onChange={f('speed')}/></label>
    <label><input type="checkbox" checked={settings.wireframe} onChange={b('wireframe')}/> wireframe</label>
    {/* mode shapes have arbitrary amplitude (M-normalised) — a um readout
        only makes sense for static load cases */}
    {result && result.type === 'static' &&
      <span>dmax = {result.dmax_um?.toExponential(2)} um</span>}
    {result && result.freq_hz &&
      <span>f = {(result.freq_hz / 1e3).toFixed(2)} kHz</span>}
  </div>
}
