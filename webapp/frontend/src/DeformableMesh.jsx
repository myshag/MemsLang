import React, { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { viridis } from './lib/viridis.js'

export default function DeformableMesh({ geometry, result, settings }) {
  const base = useMemo(() => Float32Array.from(geometry.positions), [geometry])
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(base.slice(), 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(base.length), 3))
    g.setIndex(geometry.indices)
    g.computeVertexNormals()
    return g
  }, [geometry, base])

  const layerColor = useMemo(() => {
    const lc = settings.layers.map(l => l.color)
    return geometry.vertexLayer.map(i => lc[i] || [0.6, 0.6, 0.6])
  }, [geometry, settings.layers])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    const amp = (result && result.animate) ? Math.sin(2 * Math.PI * settings.speed * t) : 1
    const s = settings.scale * amp
    const pos = geom.attributes.position.array
    const col = geom.attributes.color.array
    const disp = result ? result.disp : null
    const mag = result ? result.fields.disp_mag : null
    for (let v = 0; v < base.length / 3; v++) {
      const k = v * 3
      if (disp) {
        pos[k] = base[k] + disp[k] * s
        pos[k+1] = base[k+1] + disp[k+1] * s
        pos[k+2] = base[k+2] + disp[k+2] * s
      } else {
        pos[k] = base[k]; pos[k+1] = base[k+1]; pos[k+2] = base[k+2]
      }
      let c = layerColor[v]
      // contour is time-independent (shows the mode's pattern);
      // only the geometry oscillates
      if (disp && mag && mag[v] > 1e-6) c = viridis(mag[v])
      col[k] = c[0]; col[k+1] = c[1]; col[k+2] = c[2]
    }
    geom.attributes.position.needsUpdate = true
    geom.attributes.color.needsUpdate = true
    geom.computeVertexNormals()
  })

  return <mesh geometry={geom}>
    <meshStandardMaterial vertexColors wireframe={settings.wireframe}
      flatShading metalness={0.1} roughness={0.8} side={THREE.DoubleSide}/>
  </mesh>
}
