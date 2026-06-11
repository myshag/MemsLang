import React from 'react'
export default function ExampleSelector({ examples, value, onChange }) {
  return <select value={value || ''} onChange={e => onChange(e.target.value)}>
    <option value="" disabled>Choose a device…</option>
    {examples.map(e => <option key={e.name} value={e.name}>{e.title}</option>)}
  </select>
}
