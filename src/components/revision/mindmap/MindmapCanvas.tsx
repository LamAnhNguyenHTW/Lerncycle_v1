'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  ConnectionLineType,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import { CustomNode } from './CustomNode';
import { useTheme } from 'next-themes';

type LearningTreeNode = {
  id: string;
  label: string;
  type: 'document' | 'topic' | 'subtopic' | 'concept' | 'objective';
  summary?: string | null;
  pageStart?: number | null;
  pageEnd?: number | null;
  confidence?: number | null;
  chunkIds: string[];
  children: LearningTreeNode[];
};

interface Props {
  tree: LearningTreeNode;
}

const nodeTypes = {
  custom: CustomNode,
};

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 250;
const nodeHeight = 100;

const getLayoutedElements = (nodes: any[], edges: any[], direction = 'LR') => {
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = direction === 'LR' ? 'left' : 'top';
    node.sourcePosition = direction === 'LR' ? 'right' : 'bottom';

    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return node;
  });

  return { nodes, edges };
};

export function MindmapCanvas({ tree }: Props) {
  const { resolvedTheme } = useTheme();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    // Flatten the tree into nodes and edges
    const initialNodes: any[] = [];
    const initialEdges: any[] = [];

    const traverse = (node: LearningTreeNode, parentId: string | null = null) => {
      initialNodes.push({
        id: node.id,
        type: 'custom',
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          summary: node.summary,
          pageStart: node.pageStart,
          pageEnd: node.pageEnd,
          chunkIds: node.chunkIds,
        },
        position: { x: 0, y: 0 }, // Position will be calculated by dagre
      });

      if (parentId) {
        initialEdges.push({
          id: `e${parentId}-${node.id}`,
          source: parentId,
          target: node.id,
          type: 'smoothstep',
          animated: true,
          style: { stroke: '#94a3b8', strokeWidth: 1.5, opacity: 0.5 },
        });
      }

      for (const child of node.children) {
        traverse(child, node.id);
      }
    };

    traverse(tree);

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
      initialNodes,
      initialEdges,
      'LR'
    );

    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [tree, setNodes, setEdges]);

  return (
    <div className="w-full h-[600px] border border-border rounded-xl bg-card overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        connectionLineType={ConnectionLineType.SmoothStep}
        fitView
        colorMode={resolvedTheme === 'dark' ? 'dark' : 'light'}
        minZoom={0.1}
      >
        <Background gap={16} />
        <Controls />
        <Panel position="top-right" className="bg-background/80 backdrop-blur-sm p-2 rounded-lg border shadow-sm text-xs text-muted-foreground mr-2 mt-2">
          Scroll to zoom, drag to pan
        </Panel>
      </ReactFlow>
    </div>
  );
}
