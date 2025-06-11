//! Pattern network construction and analysis

use crate::relationship_discovery::PatternCorrelation;
use crate::types::Pattern;
use crate::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Network of pattern relationships
#[derive(Debug, Clone)]
pub struct PatternNetwork {
    /// Nodes (patterns) in the network
    pub nodes: Vec<PatternNode>,

    /// Edges (relationships) between patterns
    pub edges: Vec<PatternEdge>,

    /// Adjacency list for efficient traversal
    adjacency: HashMap<String, Vec<usize>>,
}

/// Node in the pattern network
#[derive(Debug, Clone)]
pub struct PatternNode {
    /// Pattern ID
    pub id: String,

    /// Pattern frequency
    pub frequency: f64,

    /// Node centrality score
    pub centrality: f64,

    /// Community/cluster ID
    pub community: Option<usize>,
}

/// Edge in the pattern network
#[derive(Debug, Clone)]
pub struct PatternEdge {
    /// Source pattern ID
    pub from: String,

    /// Target pattern ID
    pub to: String,

    /// Edge weight (correlation strength)
    pub weight: f64,

    /// Edge type
    pub edge_type: EdgeType,
}

/// Type of relationship between patterns
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    /// Positive correlation
    Positive,
    /// Negative correlation
    Negative,
    /// Causal relationship
    Causal,
    /// Hierarchical relationship
    Hierarchical,
}

impl PatternNetwork {
    /// Build network from patterns and correlations
    pub fn build(patterns: &[Pattern], correlations: &[PatternCorrelation]) -> Result<Self> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut adjacency: HashMap<String, Vec<usize>> = HashMap::new();

        // Create nodes
        for pattern in patterns {
            nodes.push(PatternNode {
                id: pattern.id.clone(),
                frequency: pattern.frequency,
                centrality: 0.0,
                community: None,
            });
            adjacency.insert(pattern.id.clone(), Vec::new());
        }

        // Create edges from significant correlations
        for (edge_idx, corr) in correlations.iter().enumerate() {
            if corr.p_value < 0.05 && corr.correlation.abs() > 0.3 {
                let edge_type =
                    if corr.correlation > 0.0 { EdgeType::Positive } else { EdgeType::Negative };

                edges.push(PatternEdge {
                    from: corr.pattern_a.clone(),
                    to: corr.pattern_b.clone(),
                    weight: corr.correlation.abs(),
                    edge_type,
                });

                // Update adjacency lists
                adjacency.get_mut(&corr.pattern_a).unwrap().push(edge_idx);
                adjacency.get_mut(&corr.pattern_b).unwrap().push(edge_idx);
            }
        }

        let mut network = Self {
            nodes,
            edges,
            adjacency,
        };

        // Compute centrality scores
        network.compute_centrality();

        // Detect communities
        network.detect_communities();

        Ok(network)
    }

    /// Compute centrality scores for all nodes
    fn compute_centrality(&mut self) {
        // Use degree centrality (normalized)
        let max_degree = self.nodes.len() - 1;

        for node in &mut self.nodes {
            let degree = self.adjacency.get(&node.id).unwrap().len();
            node.centrality = degree as f64 / max_degree as f64;
        }
    }

    /// Detect communities using simple clustering
    fn detect_communities(&mut self) {
        let mut community_id = 0;
        let mut visited = HashSet::new();

        let node_ids: Vec<String> = self.nodes.iter().map(|n| n.id.clone()).collect();

        for node_id in node_ids {
            if visited.contains(&node_id) {
                continue;
            }

            // BFS to find connected component
            let mut queue = VecDeque::new();
            queue.push_back(&node_id);

            while let Some(current_id) = queue.pop_front() {
                if visited.contains(current_id) {
                    continue;
                }
                visited.insert(current_id.clone());

                // Assign community
                if let Some(node_idx) = self.nodes.iter().position(|n| n.id == *current_id) {
                    self.nodes[node_idx].community = Some(community_id);
                }

                // Add neighbors
                if let Some(edge_indices) = self.adjacency.get(current_id) {
                    for &edge_idx in edge_indices {
                        let edge = &self.edges[edge_idx];
                        let neighbor = if edge.from == *current_id { &edge.to } else { &edge.from };

                        if !visited.contains(neighbor) {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            community_id += 1;
        }
    }

    /// Find shortest path between two patterns
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(edge_indices) = self.adjacency.get(&current) {
                for &edge_idx in edge_indices {
                    let edge = &self.edges[edge_idx];
                    let neighbor = if edge.from == current { &edge.to } else { &edge.from };

                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        parent.insert(neighbor.clone(), current.clone());

                        if neighbor == to {
                            // Reconstruct path
                            let mut path = vec![to.to_string()];
                            let mut current = to;

                            while let Some(p) = parent.get(current) {
                                path.push(p.clone());
                                current = p;
                            }

                            path.reverse();
                            return Some(path);
                        }

                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    /// Get most central patterns
    pub fn get_hub_patterns(&self, top_n: usize) -> Vec<&PatternNode> {
        let mut sorted_nodes: Vec<&PatternNode> = self.nodes.iter().collect();
        sorted_nodes.sort_by(|a, b| {
            b.centrality.partial_cmp(&a.centrality).unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_nodes.into_iter().take(top_n).collect()
    }

    /// Get patterns in same community
    pub fn get_community_members(&self, pattern_id: &str) -> Vec<&PatternNode> {
        let community = self.nodes.iter().find(|n| n.id == pattern_id).and_then(|n| n.community);

        if let Some(comm_id) = community {
            self.nodes.iter().filter(|n| n.community == Some(comm_id)).collect()
        } else {
            vec![]
        }
    }

    /// Analyze network statistics
    pub fn get_statistics(&self) -> NetworkStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        let avg_degree =
            if node_count > 0 { 2.0 * edge_count as f64 / node_count as f64 } else { 0.0 };

        let community_count = self.nodes.iter().filter_map(|n| n.community).max().unwrap_or(0) + 1;

        let positive_edges =
            self.edges.iter().filter(|e| e.edge_type == EdgeType::Positive).count();

        let negative_edges =
            self.edges.iter().filter(|e| e.edge_type == EdgeType::Negative).count();

        NetworkStatistics {
            node_count,
            edge_count,
            avg_degree,
            community_count,
            positive_edges,
            negative_edges,
        }
    }
}

/// Network statistics
#[derive(Debug)]
pub struct NetworkStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub community_count: usize,
    pub positive_edges: usize,
    pub negative_edges: usize,
}
