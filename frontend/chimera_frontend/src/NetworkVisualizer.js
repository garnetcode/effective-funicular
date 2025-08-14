import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Bounds } from '@react-three/drei';
import * as THREE from 'three';
import { forceSimulation, forceLink, forceManyBody, forceCenter } from 'd3-force';

const GNGNode = ({ node }) => {
  const meshRef = useRef();
  const scale = useMemo(() => 0.1 + Math.min(node.error * 10, 0.4), [node.error]);

  // The node object from d3-force has x, y, and z properties.
  const position = useMemo(() => new THREE.Vector3(node.x, node.y, node.z), [node.x, node.y, node.z]);

  return (
    <group position={position}>
      <mesh ref={meshRef} scale={[scale, scale, scale]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial color={node.error > 0.1 ? 'red' : 'orange'} />
      </mesh>
      <Text
        position={[0, scale + 0.2, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`ID: ${node.id}`}
      </Text>
    </group>
  );
};

const GNGEdge = ({ edge }) => {
  // The edge object from d3-force has source and target properties, which are node objects.
  const start = useMemo(() => new THREE.Vector3(edge.source.x, edge.source.y, edge.source.z), [edge.source]);
  const end = useMemo(() => new THREE.Vector3(edge.target.x, edge.target.y, edge.target.z), [edge.target]);
  const points = useMemo(() => [start, end], [start, end]);
  const lineGeometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial color="gray" linewidth={1} />
    </line>
  );
};

const ForceGraph = ({ graphData }) => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  const simulationRef = useRef();

  useEffect(() => {
    const gngNodes = graphData?.gng_state?.nodes || {};
    const gngEdges = graphData?.gng_state?.edges || [];

    const nodeArray = Object.entries(gngNodes).map(([id, data]) => {
      // Initialize position from weight vector for stable layout, falling back to random
      const [x, y, z] = data.weight ? new THREE.Vector3(...data.weight).multiplyScalar(5).toArray() : [Math.random(), Math.random(), Math.random()];
      return {
        id: parseInt(id, 10),
        ...data,
        x,
        y,
        z,
      };
    });

    // d3-force expects links to reference node objects or ids.
    // We use IDs and tell the forceLink to look up nodes by their 'id' field.
    const edgeArray = gngEdges.map(([source, target]) => ({
      source: parseInt(source, 10),
      target: parseInt(target, 10),
    }));

    setNodes(nodeArray);
    setEdges(edgeArray);

    if (nodeArray.length > 0) {
      simulationRef.current = forceSimulation(nodeArray, 3) // Create a 3D simulation
        .force("link", forceLink(edgeArray).id(d => d.id).distance(1).strength(0.1))
        .force("charge", forceManyBody().strength(-10))
        .force("center", forceCenter());
    }

    return () => {
      simulationRef.current?.stop();
    };
  }, [graphData]);

  useFrame(() => {
    if (simulationRef.current) {
      simulationRef.current.tick();
      // The simulation modifies the node array in place.
      // We trigger a re-render by creating a new array reference.
      setNodes(prevNodes => [...prevNodes]);
    }
  });

  return (
    <>
      {nodes.map((node) => (
        <GNGNode key={node.id} node={node} />
      ))}
      {edges.map((edge, index) => (
        <GNGEdge key={index} edge={edge} />
      ))}
    </>
  );
};


const NetworkVisualizer = ({ graphData }) => {
  if (!graphData || !graphData.gng_state) {
    return <div style={{ color: 'white', margin: '20px' }}>Loading graph data or no data available...</div>;
  }

  return (
    <Canvas camera={{ position: [0, 5, 15], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Bounds fit clip observe>
        <ForceGraph graphData={graphData} />
      </Bounds>
      <OrbitControls />
    </Canvas>
  );
};

export default NetworkVisualizer;
