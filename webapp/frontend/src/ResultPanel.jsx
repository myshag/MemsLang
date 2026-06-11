import React from 'react'
export default function ResultPanel({ results, active, onSelect }) {
  return <div style={{minWidth:220}}>
    <h4>Results</h4>
    <div onClick={() => onSelect(-1)}
      style={{cursor:'pointer', padding:'4px 0', fontWeight: active===-1?'bold':'normal'}}>
      Undeformed
    </div>
    {results.map((r, i) =>
      <div key={i} onClick={() => onSelect(i)}
        style={{cursor:'pointer', padding:'4px 0', fontWeight: active===i?'bold':'normal'}}>
        {r.label}{r.freq_hz ? ` — ${(r.freq_hz/1e3).toFixed(2)} kHz` : ''}
      </div>)}
  </div>
}
