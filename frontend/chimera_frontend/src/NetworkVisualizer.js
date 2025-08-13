import React, { useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

const GNGNode = ({ position, id, error }) => {
  const meshRef = useRef();
  const scale = useMemo(() => 0.1 + Math.min(error * 10, 0.4), [error]);

  return (
    <group position={position}>
      <mesh ref={meshRef} scale={[scale, scale, scale]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial color={error > 0.1 ? 'red' : 'orange'} />
      </mesh>
      <Text
        position={[0, scale + 0.2, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`ID: ${id}`}
      </Text>
    </group>
  );
};

const GNGEdge = ({ start, end }) => {
  const points = useMemo(() => [new THREE.Vector3(...start), new THREE.Vector3(...end)], [start, end]);
  const lineGeometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial color="gray" linewidth={1} />
    </line>
  );
};


const NetworkVisualizer = ({ graphData }) => {
  // CORRECTED: Wrap dependency initializations in useMemo to ensure stable references.
  const nodes = useMemo(() => graphData?.gng_state?.nodes || {}, [graphData]);
  const edges = useMemo(() => graphData?.gng_state?.edges || [], [graphData]);

  const nodeMap = useMemo(() => {
    const map = new Map();
    Object.entries(nodes).forEach(([id, data]) => {
      const position = new THREE.Vector3(...data.weight).multiplyScalar(5);
      map.set(parseInt(id, 10), { ...data, position });
    });
    return map;
  }, [nodes]);

  if (!graphData || !graphData.gng_state) {
    return <div style={{ color: 'white', margin: '20px' }}>Loading graph data or no data available...</div>;
  }

  return (
    <Canvas camera={{ position: [0, 5, 15], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />

      {Array.from(nodeMap.entries()).map(([id, node]) => (
        <GNGNode key={id} id={id} position={node.position} error={node.error} />
      ))}

      {edges.map((edge, index) => {
        const [sourceId, targetId] = edge;
        const sourceNode = nodeMap.get(sourceId);
        const targetNode = nodeMap.get(targetId);
        if (!sourceNode || !targetNode) return null;
        return (
          <GNGEdge key={index} start={sourceNode.position.toArray()} end={targetNode.position.toArray()} />
        );
      })}

      <OrbitControls />
    </Canvas>
  );
};

export default NetworkVisualizer;
