import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Bounds } from '@react-three/drei'
import DeformableMesh from './DeformableMesh.jsx'

export default function DeviceCanvas({ bundle, result, settings }) {
  return <Canvas camera={{ position: [400, 400, 600], far: 100000 }}
                 style={{ background: '#0b0d12' }}>
    <ambientLight intensity={0.6}/>
    <directionalLight position={[1, 2, 3]} intensity={1.0}/>
    <directionalLight position={[-2, -1, -1]} intensity={0.3}/>
    <Bounds fit clip observe margin={1.2}>
      <DeformableMesh geometry={bundle.geometry} result={result}
        settings={{ ...settings, layers: bundle.meta.layers }}/>
    </Bounds>
    <OrbitControls makeDefault/>
  </Canvas>
}
