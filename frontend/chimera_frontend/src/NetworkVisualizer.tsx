import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Bounds, Line } from '@react-three/drei';
import * as THREE from 'three';
import { forceSimulation, forceLink, forceManyBody, forceCenter, Simulation, SimulationNodeDatum, SimulationLinkDatum } from 'd3-force';

// --- Type Definitions ---

export interface GraphData {
  nodes: any[];
  edges: any[];
}

interface NodeDatum extends SimulationNodeDatum {
  id: number;
  error: number;
  utility: number;
  weight?: number[];
  x?: number;
  y?: number;
  z?: number;
}

interface EdgeDatum extends SimulationLinkDatum<NodeDatum> {
  // source and target can be string | number | NodeDatum
  // We will narrow it down in the component
}

// --- Prop Type Definitions ---

interface GNGNodeProps {
  node: NodeDatum;
}

interface GNGEdgeProps {
  edge: EdgeDatum;
}

interface ForceGraphProps {
  graphData: any; // The raw graph data from the API
}

interface NetworkVisualizerProps {
  graphData: any;
}


const GNGNode: React.FC<GNGNodeProps> = ({ node }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [isHovered, setIsHovered] = useState(false);

  const scale = useMemo(() => 0.1 + Math.min(node.utility * 0.5, 0.4), [node.utility]);
  const color = useMemo(() => {
    const utilityNormalized = Math.min(node.utility / 10, 1); // Normalize utility
    return new THREE.Color().setHSL(0.6, 1.0, 0.5 + utilityNormalized * 0.4); // Blue to Cyan/White
  }, [node.utility]);

  // The node object from d3-force has x, y, and z properties.
  const position = useMemo(() => new THREE.Vector3(node.x, node.y, node.z), [node.x, node.y, node.z]);

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        scale={isHovered ? scale * 1.5 : scale}
        onPointerOver={() => setIsHovered(true)}
        onPointerOut={() => setIsHovered(false)}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={isHovered ? 0.5 : 0.1} toneMapped={false} />
      </mesh>
      {isHovered && (
        <Text
          position={[0, scale * 1.5 + 0.2, 0]}
          fontSize={0.15}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.005}
          outlineColor="black"
        >
          {`ID: ${node.id}\nUtility: ${node.utility.toFixed(2)}\nError: ${node.error.toFixed(2)}`}
        </Text>
      )}
    </group>
  );
};

const GNGEdge: React.FC<GNGEdgeProps> = ({ edge }) => {
  const sourceNode = edge.source as NodeDatum;
  const targetNode = edge.target as NodeDatum;

  // The edge object from d3-force has source and target properties, which are node objects.
  const start = useMemo(() => new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z), [sourceNode.x, sourceNode.y, sourceNode.z]);
  const end = useMemo(() => new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z), [targetNode.x, targetNode.y, targetNode.z]);

  return (
    <Line
      points={[start, end]}
      color="#00ffff" // A cyan color for a futuristic look
      lineWidth={1}
      dashed={false}
    />
  );
};

const ForceGraph: React.FC<ForceGraphProps> = ({ graphData }) => {
  const [nodes, setNodes] = useState<NodeDatum[]>([]);
  const [edges, setEdges] = useState<EdgeDatum[]>([]);

  const simulationRef = useRef<Simulation<NodeDatum, EdgeDatum> | null>(null);

  useEffect(() => {
    const gngNodes = graphData?.nodes || {};
    const gngEdges: [number, number][] = graphData?.edges || [];

    const nodeArray: NodeDatum[] = Object.entries(gngNodes).map(([id, data]: [string, any]) => {
      // Initialize position from weight vector for stable layout, falling back to random
      const [x, y, z] = data.weight ? new THREE.Vector3(...data.weight).multiplyScalar(5).toArray() : [Math.random(), Math.random(), Math.random()];
      return {
        id: parseInt(id, 10),
        error: data.error || 0,
        utility: data.utility || 0,
        weight: data.weight,
        x,
        y,
        z,
      };
    });

    // d3-force expects links to reference node objects or ids.
    // We use IDs and tell the forceLink to look up nodes by their 'id' field.
    const edgeArray = gngEdges.map(([source, target]) => ({
      source: source,
      target: target,
    }));

    setNodes(nodeArray);
    setEdges(edgeArray);

    if (nodeArray.length > 0) {
      simulationRef.current = forceSimulation(nodeArray)
        .force("link", forceLink<NodeDatum, EdgeDatum>(edgeArray).id(d => d.id).distance(1).strength(0.1))
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


const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ graphData }) => {
  if (!graphData || !graphData.nodes) {
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
