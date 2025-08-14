import React, { useMemo, useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import SpriteText from 'three-spritetext';

const NetworkVisualizer = ({ graphData }) => {
  const fgRef = useRef();

  // Memoize the processed graph data to prevent unnecessary recalculations
  const processedData = useMemo(() => {
    if (!graphData || !graphData.gng_state || !graphData.gng_state.nodes) {
      return { nodes: [], links: [] };
    }

    const gngNodes = graphData.gng_state.nodes;
    const gngEdges = graphData.gng_state.edges;

    const nodes = Object.entries(gngNodes).map(([id, node]) => ({
      id: id,
      ...node,
      // You can add more properties here for styling, e.g., color based on error
      color: node.error > 0.1 ? 'red' : 'orange',
      val: 0.1 + Math.min(node.error * 10, 0.4) // Node size based on error
    }));

    const links = gngEdges.map(([source, target]) => ({
      source: source,
      target: target,
    }));

    return { nodes, links };
  }, [graphData]);

  useEffect(() => {
    // Zoom to fit all nodes on initial load or data change
    if (fgRef.current && processedData.nodes.length > 0) {
      fgRef.current.zoomToFit(400, 100);
    }
  }, [processedData]);


  if (processedData.nodes.length === 0) {
    return <div style={{ color: 'white', margin: '20px' }}>Loading graph data or no data available...</div>;
  }

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={processedData}
      nodeAutoColorBy="group" // Example: color by a 'group' property if you add one
      nodeThreeObject={node => {
        const sprite = new SpriteText(node.id);
        sprite.color = 'white';
        sprite.textHeight = 8;
        return sprite;
      }}
      nodeThreeObjectExtend={true}
      linkWidth={1}
      linkColor={() => 'rgba(128, 128, 128, 0.5)'}
      backgroundColor="rgba(0,0,0,0)"
    />
  );
};

export default NetworkVisualizer;
