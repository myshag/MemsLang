import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Bounds } from '@react-three/drei'
import DeformableMesh from './DeformableMesh.jsx'

export default function DeviceCanvas({ bundle, result, settings }) {
  return <Canvas camera={{ position: [350, 700, 450], far: 100000 }}
                 style={{ background: '#0b0d12' }}>
    <ambientLight intensity={0.6}/>
    <directionalLight position={[1, 2, 3]} intensity={1.0}/>
    <directionalLight position={[-2, -1, -1]} intensity={0.3}/>
    <Bounds fit clip observe margin={1.2}>
      {/* soidlc is Z-up (wafer normal = +z); three.js is Y-up */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <DeformableMesh geometry={bundle.geometry} result={result}
          settings={{ ...settings, layers: bundle.meta.layers }}/>
      </group>
    </Bounds>
    <OrbitControls makeDefault/>
  </Canvas>
}
