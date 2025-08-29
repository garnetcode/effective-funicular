import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Bounds, Line } from '@react-three/drei';
import * as THREE from 'three';
import { forceSimulation, forceLink, forceManyBody, forceCenter, Simulation, SimulationNodeDatum, SimulationLinkDatum, ForceLink } from 'd3-force';

// --- Type Definitions ---

export interface GraphData {
  nodes: any[];
  edges: any[];
}

interface NodeDatum extends SimulationNodeDatum {
  id: string;
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


const GNGNode: React.FC<GNGNodeProps> = React.memo(({ node }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [isHovered, setIsHovered] = useState(false);

  const scale = useMemo(() => 0.1 + Math.min(node.utility * 0.5, 0.4), [node.utility]);
  const color = useMemo(() => {
    const utilityNormalized = Math.min(node.utility / 20, 1);
    const hue = 0.75 - (utilityNormalized * (0.75 - 0.33));
    const saturation = 0.9;
    const lightness = 0.6;
    return new THREE.Color().setHSL(hue, saturation, lightness);
  }, [node.utility]);

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
});

const GNGEdge: React.FC<GNGEdgeProps> = React.memo(({ edge }) => {
  const sourceNode = edge.source as NodeDatum;
  const targetNode = edge.target as NodeDatum;

  const start = useMemo(() => new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z), [sourceNode.x, sourceNode.y, sourceNode.z]);
  const end = useMemo(() => new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z), [targetNode.x, targetNode.y, targetNode.z]);

  return (
    <Line
      points={[start, end]}
      color="#00ffff"
      lineWidth={1}
      dashed={false}
    />
  );
});

const ForceGraph: React.FC<ForceGraphProps> = ({ graphData }) => {
  const nodesRef = useRef<NodeDatum[]>([]);
  const [, setRenderTrigger] = useState(0);

  const simulationRef = useRef<Simulation<NodeDatum, EdgeDatum>>(
    forceSimulation<NodeDatum, EdgeDatum>()
      .force("link", forceLink<NodeDatum, EdgeDatum>().id((d: any) => d.id).distance(1).strength(0.1))
      .force("charge", forceManyBody().strength(-10))
      .force("center", forceCenter())
  );

  useEffect(() => {
    const simulation = simulationRef.current;
    const newNodesData = graphData?.nodes || {};
    const newEdgesData = graphData?.edges || [];

    const existingNodesMap = new Map(nodesRef.current.map(node => [node.id, node]));
    const updatedNodes: NodeDatum[] = [];

    // Update existing nodes and add new ones
    Object.entries(newNodesData).forEach(([id, data]: [string, any]) => {
      const existingNode = existingNodesMap.get(id);
      if (existingNode) {
        // Update existing node properties
        existingNode.utility = data.utility || 0;
        existingNode.error = data.error || 0;
        updatedNodes.push(existingNode);
      } else {
        // Add new node
        const [x, y, z] = data.weight ? new THREE.Vector3(...data.weight).multiplyScalar(5).toArray() : [Math.random(), Math.random(), Math.random()];
        updatedNodes.push({ id, ...data, x, y, z });
      }
    });

    nodesRef.current = updatedNodes;

    // Update simulation
    simulation.nodes(nodesRef.current);
    (simulation.force("link") as ForceLink<NodeDatum, EdgeDatum>)?.links(newEdgesData);
    simulation.alpha(0.3).restart(); // Reheat the simulation

  }, [graphData]);

  useFrame(() => {
    // The simulation runs continuously and modifies the node positions in the ref
    // We just need to trigger a re-render to see the changes
    setRenderTrigger(r => r + 1);
  });

  return (
    <>
      {nodesRef.current.map((node) => (
        <GNGNode key={node.id} node={node} />
      ))}
      {nodesRef.current.length > 0 && (simulationRef.current.force("link") as ForceLink<NodeDatum, EdgeDatum>)?.links().map((edge: EdgeDatum, index: number) => (
        <GNGEdge key={index} edge={edge} />
      ))}
    </>
  );
};


const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ graphData }) => {
  if (!graphData || !graphData.nodes) {
    return <div style={{ color: 'white', margin: '20px' }}>Loading graph data or no data available...</div>;
  }

  const nodeCount = Object.keys(graphData.nodes).length;
  const edgeCount = graphData.edges.length;

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div className="graph-legend">
        <div>Nodes: {nodeCount}</div>
        <div>Edges: {edgeCount}</div>
      </div>
      <Canvas camera={{ position: [0, 5, 15], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <Bounds fit clip observe>
          <ForceGraph graphData={graphData} />
        </Bounds>
        <OrbitControls />
      </Canvas>
    </div>
  );
};

export default NetworkVisualizer;
