from collections import defaultdict
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from .foreminer_aux import *

# Try to import advanced graph libraries
try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class GraphAnalyzer(AnalysisStrategy):
    """SOTA graph analysis for discovering network structures and relationships in data"""

    @property
    def name(self) -> str:
        return "graph_analysis"

    def __init__(self):
        # Performance and analysis thresholds
        self.max_nodes = 10000          # Maximum nodes for full analysis
        self.min_edge_weight = 0.1      # Minimum edge weight threshold
        self.correlation_threshold = 0.3 # Minimum correlation for edge creation
        self.max_edges_per_node = 50    # Maximum edges per node (performance)
        self.sample_threshold = 5000    # Subsample large datasets
        
        # Graph construction methods
        self.graph_methods = [
            "correlation_network",
            "knn_network", 
            "threshold_network",
            "mutual_information_network",
            "distance_network"
        ]

    # --------------------------- Graph Construction ---------------------------
    def _construct_correlation_network(self, data: pd.DataFrame, threshold: float = None) -> nx.Graph:
        """Build network based on correlation between variables/samples"""
        if threshold is None:
            threshold = self.correlation_threshold
        
        # Compute correlation matrix
        corr_matrix = data.corr().abs()
        
        # Create graph
        G = nx.Graph()
        nodes = list(corr_matrix.columns)
        G.add_nodes_from(nodes)
        
        # Add edges based on correlation threshold
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val >= threshold:
                    G.add_edge(node1, node2, weight=corr_val, type='correlation')
        
        return G

    def _construct_knn_network(self, data: pd.DataFrame, k: int = None) -> nx.Graph:
        """Build k-nearest neighbor network"""
        if k is None:
            k = min(10, max(3, len(data.columns) // 3))
        
        # Compute pairwise distances
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.T)  # Transpose for variable relationships
        distances = euclidean_distances(data_scaled)
        
        # Create graph
        G = nx.Graph()
        nodes = list(data.columns)
        G.add_nodes_from(nodes)
        
        # Add k-nearest neighbor edges
        for i, node1 in enumerate(nodes):
            # Get k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:k+1]
            for j in neighbor_indices:
                node2 = nodes[j]
                distance = distances[i, j]
                weight = 1 / (1 + distance)  # Convert distance to similarity
                G.add_edge(node1, node2, weight=weight, type='knn')
        
        return G

    def _construct_threshold_network(self, data: pd.DataFrame, method: str = 'euclidean') -> nx.Graph:
        """Build network using distance threshold"""
        
        # Compute pairwise similarities/distances
        if method == 'euclidean':
            distances = euclidean_distances(data.T)
            # Convert to similarities
            similarities = 1 / (1 + distances)
        elif method == 'cosine':
            similarities = cosine_similarity(data.T)
        else:
            # Default to correlation
            similarities = data.corr().abs().values
        
        # Determine adaptive threshold
        threshold = np.percentile(similarities[similarities > 0], 75)
        
        # Create graph
        G = nx.Graph()
        nodes = list(data.columns)
        G.add_nodes_from(nodes)
        
        # Add edges above threshold
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                similarity = similarities[i, j]
                if similarity >= threshold:
                    G.add_edge(node1, node2, weight=similarity, type='threshold')
        
        return G

    def _construct_mutual_information_network(self, data: pd.DataFrame) -> nx.Graph:
        """Build network based on mutual information"""
        from sklearn.feature_selection import mutual_info_regression

        # Subsample if too large
        if len(data) > self.sample_threshold:
            sample_data = data.sample(self.sample_threshold, random_state=42)
        else:
            sample_data = data
        
        # Compute pairwise mutual information
        columns = list(sample_data.columns)
        mi_matrix = np.zeros((len(columns), len(columns)))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    try:
                        mi_val = mutual_info_regression(
                            sample_data[col1].values.reshape(-1, 1),
                            sample_data[col2].values,
                            random_state=42
                        )[0]
                        mi_matrix[i, j] = mi_val
                    except:
                        mi_matrix[i, j] = 0
        
        # Normalize MI values
        if mi_matrix.max() > 0:
            mi_matrix = mi_matrix / mi_matrix.max()
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(columns)
        
        # Add edges based on MI threshold
        threshold = np.percentile(mi_matrix[mi_matrix > 0], 70) if (mi_matrix > 0).any() else 0
        
        for i, node1 in enumerate(columns):
            for j, node2 in enumerate(columns[i+1:], i+1):
                mi_val = mi_matrix[i, j]
                if mi_val >= threshold:
                    G.add_edge(node1, node2, weight=mi_val, type='mutual_info')
        
        return G

    def _construct_sample_network(self, data: pd.DataFrame, method: str = 'correlation') -> nx.Graph:
        """Build network between samples (rows) instead of features"""
        
        # Subsample if too many samples
        if len(data) > 1000:
            sample_indices = np.random.choice(len(data), 1000, replace=False)
            sample_data = data.iloc[sample_indices]
            node_names = [f"Sample_{i}" for i in sample_indices]
        else:
            sample_data = data
            node_names = [f"Sample_{i}" for i in range(len(data))]
        
        # Compute similarities between samples
        if method == 'correlation':
            similarities = np.corrcoef(sample_data.values)
            similarities = np.abs(similarities)  # Use absolute correlation
        elif method == 'cosine':
            similarities = cosine_similarity(sample_data.values)
        else:  # euclidean
            distances = euclidean_distances(sample_data.values)
            similarities = 1 / (1 + distances)
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(node_names)
        
        # Adaptive threshold
        threshold = np.percentile(similarities[similarities > 0], 80)
        
        # Add edges
        for i in range(len(node_names)):
            for j in range(i+1, len(node_names)):
                similarity = similarities[i, j]
                if similarity >= threshold:
                    G.add_edge(node_names[i], node_names[j], 
                             weight=similarity, type=f'sample_{method}')
        
        return G

    # --------------------------- Network Analysis ---------------------------
    def _analyze_network_topology(self, G: nx.Graph) -> Dict[str, Any]:
        """Comprehensive network topology analysis"""
        if G.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        topology = {}
        
        # Basic properties
        topology["n_nodes"] = G.number_of_nodes()
        topology["n_edges"] = G.number_of_edges()
        topology["density"] = nx.density(G)
        topology["is_connected"] = nx.is_connected(G)
        
        if G.number_of_edges() == 0:
            return topology
        
        # Connectivity
        if nx.is_connected(G):
            topology["diameter"] = nx.diameter(G)
            topology["radius"] = nx.radius(G)
            topology["average_path_length"] = nx.average_shortest_path_length(G)
        else:
            # For disconnected graphs
            components = list(nx.connected_components(G))
            topology["n_components"] = len(components)
            topology["largest_component_size"] = len(max(components, key=len))
            
            # Analyze largest component
            largest_component = G.subgraph(max(components, key=len))
            if largest_component.number_of_edges() > 0:
                topology["largest_component_diameter"] = nx.diameter(largest_component)
                topology["largest_component_avg_path_length"] = nx.average_shortest_path_length(largest_component)
        
        # Clustering
        topology["average_clustering"] = nx.average_clustering(G)
        topology["transitivity"] = nx.transitivity(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            topology["degree_stats"] = {
                "mean": float(np.mean(degrees)),
                "std": float(np.std(degrees)),
                "min": int(min(degrees)),
                "max": int(max(degrees)),
                "median": float(np.median(degrees))
            }
        
        # Assortativity
        try:
            topology["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
        except:
            topology["degree_assortativity"] = None
        
        # Small world properties
        try:
            # Compare with random graph
            n, m = G.number_of_nodes(), G.number_of_edges()
            if m > 0 and n > 1:
                random_clustering = (2 * m) / (n * (n - 1))
                random_path_length = np.log(n) / np.log(2 * m / n) if m > 0 else float('inf')
                
                actual_clustering = topology["average_clustering"]
                actual_path_length = topology.get("average_path_length", float('inf'))
                
                if random_clustering > 0 and random_path_length < float('inf'):
                    small_world_sigma = (actual_clustering / random_clustering) / (actual_path_length / random_path_length)
                    topology["small_world_sigma"] = small_world_sigma
                    topology["is_small_world"] = small_world_sigma > 1
        except:
            pass
        
        return topology

    def _detect_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """Advanced community detection using multiple algorithms"""
        if G.number_of_nodes() < 3:
            return {"error": "Too few nodes for community detection"}
        
        communities = {}
        
        # Method 1: Louvain (if available)
        if HAS_LOUVAIN:
            try:
                partition = community_louvain.best_partition(G, random_state=42)
                communities["louvain"] = {
                    "partition": partition,
                    "n_communities": len(set(partition.values())),
                    "modularity": community_louvain.modularity(partition, G),
                    "method": "Louvain"
                }
            except Exception as e:
                communities["louvain"] = {"error": str(e)}
        
        # Method 2: Girvan-Newman
        try:
            if G.number_of_edges() > 0 and G.number_of_nodes() <= 100:  # Limit for performance
                gn_communities = list(nx.community.girvan_newman(G))
                if gn_communities:
                    # Take the partition with reasonable number of communities
                    best_partition = None
                    for partition in gn_communities:
                        if 2 <= len(partition) <= G.number_of_nodes() // 2:
                            best_partition = partition
                            break
                    
                    if best_partition:
                        # Convert to node->community dict
                        partition_dict = {}
                        for i, community in enumerate(best_partition):
                            for node in community:
                                partition_dict[node] = i
                        
                        communities["girvan_newman"] = {
                            "partition": partition_dict,
                            "n_communities": len(best_partition),
                            "modularity": nx.community.modularity(G, best_partition),
                            "method": "Girvan-Newman"
                        }
        except Exception as e:
            communities["girvan_newman"] = {"error": str(e)}
        
        # Method 3: Spectral clustering (using scikit-learn)
        try:
            if G.number_of_nodes() >= 4:
                # Convert to adjacency matrix
                adj_matrix = nx.adjacency_matrix(G).toarray()
                
                # Try different numbers of clusters
                best_n_clusters = 2
                best_score = -1
                
                for n_clusters in range(2, min(8, G.number_of_nodes() // 2)):
                    try:
                        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
                        labels = spectral.fit_predict(adj_matrix)
                        
                        # Convert to partition dict
                        partition = {node: int(labels[i]) for i, node in enumerate(G.nodes())}
                        
                        # Calculate modularity
                        partition_sets = defaultdict(set)
                        for node, cluster in partition.items():
                            partition_sets[cluster].add(node)
                        
                        modularity = nx.community.modularity(G, partition_sets.values())
                        
                        if modularity > best_score:
                            best_score = modularity
                            best_n_clusters = n_clusters
                            best_partition = partition
                    except:
                        continue
                
                if best_score > -1:
                    communities["spectral"] = {
                        "partition": best_partition,
                        "n_communities": best_n_clusters,
                        "modularity": best_score,
                        "method": "Spectral Clustering"
                    }
        except Exception as e:
            communities["spectral"] = {"error": str(e)}
        
        # Method 4: Label propagation
        try:
            lp_communities = list(nx.community.label_propagation_communities(G))
            if lp_communities:
                partition_dict = {}
                for i, community in enumerate(lp_communities):
                    for node in community:
                        partition_dict[node] = i
                
                communities["label_propagation"] = {
                    "partition": partition_dict,
                    "n_communities": len(lp_communities),
                    "modularity": nx.community.modularity(G, lp_communities),
                    "method": "Label Propagation"
                }
        except Exception as e:
            communities["label_propagation"] = {"error": str(e)}
        
        return communities

    def _centrality_analysis(self, G: nx.Graph) -> Dict[str, Any]:
        """Comprehensive centrality analysis"""
        if G.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        centralities = {}
        
        # Degree centrality
        try:
            degree_cent = nx.degree_centrality(G)
            centralities["degree"] = {
                "values": degree_cent,
                "top_nodes": sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                "description": "Measures local connectivity"
            }
        except Exception as e:
            centralities["degree"] = {"error": str(e)}
        
        # Closeness centrality (only for connected components)
        try:
            if nx.is_connected(G):
                closeness_cent = nx.closeness_centrality(G)
                centralities["closeness"] = {
                    "values": closeness_cent,
                    "top_nodes": sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                    "description": "Measures how close a node is to all other nodes"
                }
            else:
                # Compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                closeness_cent = nx.closeness_centrality(subgraph)
                centralities["closeness"] = {
                    "values": closeness_cent,
                    "top_nodes": sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                    "description": "Measures how close a node is to all other nodes (largest component only)",
                    "component_size": len(largest_cc)
                }
        except Exception as e:
            centralities["closeness"] = {"error": str(e)}
        
        # Betweenness centrality
        try:
            if G.number_of_nodes() <= 500:  # Limit for performance
                betweenness_cent = nx.betweenness_centrality(G)
                centralities["betweenness"] = {
                    "values": betweenness_cent,
                    "top_nodes": sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                    "description": "Measures how often a node lies on shortest paths"
                }
            else:
                # Use approximation for large graphs
                betweenness_cent = nx.betweenness_centrality(G, k=100)
                centralities["betweenness"] = {
                    "values": betweenness_cent,
                    "top_nodes": sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                    "description": "Measures how often a node lies on shortest paths (approximated)"
                }
        except Exception as e:
            centralities["betweenness"] = {"error": str(e)}
        
        # Eigenvector centrality
        try:
            if G.number_of_edges() > 0:
                eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
                centralities["eigenvector"] = {
                    "values": eigenvector_cent,
                    "top_nodes": sorted(eigenvector_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                    "description": "Measures influence based on connections to influential nodes"
                }
        except Exception as e:
            centralities["eigenvector"] = {"error": str(e)}
        
        # PageRank
        try:
            pagerank_cent = nx.pagerank(G, alpha=0.85)
            centralities["pagerank"] = {
                "values": pagerank_cent,
                "top_nodes": sorted(pagerank_cent.items(), key=lambda x: x[1], reverse=True)[:5],
                "description": "Google's PageRank algorithm adapted for general networks"
            }
        except Exception as e:
            centralities["pagerank"] = {"error": str(e)}
        
        return centralities

    def _network_embedding_analysis(self, G: nx.Graph) -> Dict[str, Any]:
        """Advanced network embedding analysis"""
        embeddings = {}
        
        # Node2Vec (if available)
        if HAS_NODE2VEC and G.number_of_nodes() >= 4:
            try:
                # Create Node2Vec model
                node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1)
                model = node2vec.fit(window=10, min_count=1, batch_words=4)
                
                # Get embeddings
                embedding_dict = {}
                for node in G.nodes():
                    embedding_dict[node] = model.wv[node].tolist()
                
                embeddings["node2vec"] = {
                    "embeddings": embedding_dict,
                    "dimensions": 64,
                    "method": "Node2Vec",
                    "description": "Deep learning-based node embeddings"
                }
                
                # Compute embedding similarities
                nodes = list(G.nodes())
                if len(nodes) >= 2:
                    embedding_matrix = np.array([model.wv[node] for node in nodes])
                    similarities = cosine_similarity(embedding_matrix)
                    
                    # Find most similar node pairs
                    similar_pairs = []
                    for i in range(len(nodes)):
                        for j in range(i+1, len(nodes)):
                            similar_pairs.append((nodes[i], nodes[j], similarities[i, j]))
                    
                    # Sort by similarity
                    similar_pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    embeddings["node2vec"]["most_similar_pairs"] = similar_pairs[:5]
                
            except Exception as e:
                embeddings["node2vec"] = {"error": str(e)}
        
        # Spectral embedding (using NetworkX)
        try:
            if G.number_of_nodes() >= 3 and G.number_of_edges() > 0:
                # Compute Laplacian eigenvectors
                laplacian_matrix = nx.laplacian_matrix(G).toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
                
                # Sort by eigenvalue
                idx = eigenvalues.argsort()
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Use first few non-zero eigenvectors for embedding
                n_dims = min(8, G.number_of_nodes() - 1)
                spectral_embedding = eigenvectors[:, 1:n_dims+1]  # Skip first (trivial) eigenvector
                
                # Convert to dict
                nodes = list(G.nodes())
                embedding_dict = {}
                for i, node in enumerate(nodes):
                    embedding_dict[node] = spectral_embedding[i].tolist()
                
                embeddings["spectral"] = {
                    "embeddings": embedding_dict,
                    "eigenvalues": eigenvalues[:n_dims+1].tolist(),
                    "dimensions": n_dims,
                    "method": "Spectral Embedding",
                    "description": "Laplacian eigenvector-based embeddings"
                }
                
        except Exception as e:
            embeddings["spectral"] = {"error": str(e)}
        
        return embeddings

    # --------------------------- Main Analysis Method ---------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig, graph_type: str = "auto", 
                target_col: str = None) -> Dict[str, Any]:
        """
        Comprehensive graph analysis of data relationships
        
        Args:
            data: DataFrame containing the data
            graph_type: Type of graph to construct ("correlation", "knn", "threshold", 
                       "mutual_info", "sample", "auto")
            target_col: Optional target column for supervised graph construction
            
        Returns:
            Dictionary containing comprehensive graph analysis results
        """
        try:
            # Data preprocessing
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            
            if numeric_data.empty:
                raise ValueError("No numeric data available for graph analysis")
            
            if numeric_data.shape[1] < 2:
                raise ValueError("Need at least 2 numeric columns for graph analysis")
            
            # Subsample if too large
            if len(numeric_data) > self.sample_threshold:
                sample_data = numeric_data.sample(self.sample_threshold, random_state=42)
                was_subsampled = True
            else:
                sample_data = numeric_data
                was_subsampled = False
            
            # Remove constant columns
            constant_cols = sample_data.columns[sample_data.std() < 1e-10]
            if len(constant_cols) > 0:
                sample_data = sample_data.drop(columns=constant_cols)
            
            if sample_data.shape[1] < 2:
                raise ValueError("Insufficient non-constant columns for graph analysis")
            
            analysis_results = {
                "data_info": {
                    "original_shape": data.shape,
                    "analyzed_shape": sample_data.shape,
                    "was_subsampled": was_subsampled,
                    "constant_columns_removed": list(constant_cols),
                    "analysis_type": "graph_network_analysis"
                },
                "graphs": {},
                "network_topology": {},
                "communities": {},
                "centralities": {},
                "embeddings": {},
                "summary": {},
                "recommendations": []
            }
            
            # Auto-detect best graph type or construct specified type
            if graph_type == "auto":
                graph_types_to_try = ["correlation", "mutual_info", "knn"]
            else:
                graph_types_to_try = [graph_type]
            
            # Construct and analyze graphs
            for gtype in graph_types_to_try:
                try:
                    # Construct graph
                    if gtype == "correlation":
                        G = self._construct_correlation_network(sample_data)
                    elif gtype == "knn":
                        G = self._construct_knn_network(sample_data)
                    elif gtype == "threshold":
                        G = self._construct_threshold_network(sample_data)
                    elif gtype == "mutual_info":
                        G = self._construct_mutual_information_network(sample_data)
                    elif gtype == "sample":
                        G = self._construct_sample_network(sample_data)
                    else:
                        continue
                    
                    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                        continue
                    
                    # Store graph info
                    analysis_results["graphs"][gtype] = {
                        "n_nodes": G.number_of_nodes(),
                        "n_edges": G.number_of_edges(),
                        "edge_types": list(set([G[u][v].get('type', 'unknown') for u, v in G.edges()])),
                        "graph_object": G  # Store for potential further analysis
                    }
                    
                    # Analyze network topology
                    topology = self._analyze_network_topology(G)
                    analysis_results["network_topology"][gtype] = topology
                    
                    # Community detection
                    communities = self._detect_communities(G)
                    analysis_results["communities"][gtype] = communities
                    
                    # Centrality analysis
                    centralities = self._centrality_analysis(G)
                    analysis_results["centralities"][gtype] = centralities
                    
                    # Network embeddings (only for best graph to save time)
                    if gtype == graph_types_to_try[0] or len(analysis_results["graphs"]) == 1:
                        embeddings = self._network_embedding_analysis(G)
                        analysis_results["embeddings"][gtype] = embeddings
                
                except Exception as e:
                    analysis_results["graphs"][gtype] = {"error": str(e)}
                    continue
            
            # Generate summary and recommendations
            self._generate_graph_summary_and_recommendations(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            return {
                "error": f"Graph analysis failed: {e}",
                "analysis_type": "graph_network_analysis"
            }

    def _generate_graph_summary_and_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate summary statistics and recommendations"""
        
        # Find best graph based on connectivity and structure
        best_graph = None
        best_score = -1
        
        for graph_type, topology in results["network_topology"].items():
            if "error" in topology:
                continue
            
            # Score graph quality
            score = 0
            
            # Connectivity bonus
            if topology.get("is_connected", False):
                score += 10
            else:
                # Partial connectivity
                n_components = topology.get("n_components", float('inf'))
                if n_components < float('inf'):
                    score += max(0, 10 - n_components)
            
            # Density bonus (not too sparse, not too dense)
            density = topology.get("density", 0)
            if 0.1 <= density <= 0.7:
                score += 5
            elif density > 0:
                score += 2
            
            # Small world bonus
            if topology.get("is_small_world", False):
                score += 5
            
            # Clustering bonus
            clustering = topology.get("average_clustering", 0)
            if clustering > 0.3:
                score += 3
            
            if score > best_score:
                best_score = score
                best_graph = graph_type
        
        # Summary
        summary = {
            "best_graph_type": best_graph,
            "graphs_constructed": list(results["graphs"].keys()),
            "successful_constructions": len([g for g in results["graphs"].values() if "error" not in g]),
            "total_attempted": len(results["graphs"])
        }
        
        if best_graph and best_graph in results["network_topology"]:
            best_topology = results["network_topology"][best_graph]
            summary.update({
                "best_graph_nodes": best_topology.get("n_nodes", 0),
                "best_graph_edges": best_topology.get("n_edges", 0),
                "best_graph_density": best_topology.get("density", 0),
                "best_graph_connected": best_topology.get("is_connected", False),
                "best_graph_clustering": best_topology.get("average_clustering", 0)
            })
        
        results["summary"] = summary
        
        # Recommendations
        recommendations = []
        
        if best_graph:
            best_topology = results["network_topology"][best_graph]
            best_communities = results["communities"].get(best_graph, {})
            best_centralities = results["centralities"].get(best_graph, {})
            
            recommendations.append(f"🏆 **Best graph structure**: {best_graph.replace('_', ' ').title()} Network")
            
            # Connectivity insights
            if best_topology.get("is_connected", False):
                recommendations.append("🔗 **Highly connected network** - variables form a cohesive system")
                
                # Path length insights
                avg_path = best_topology.get("average_path_length", 0)
                if avg_path <= 3:
                    recommendations.append("⚡ **Short path lengths** - information flows efficiently through network")
                elif avg_path > 5:
                    recommendations.append("🌉 **Long path lengths** - some variables are distantly related")
            else:
                n_components = best_topology.get("n_components", 0)
                if n_components > 1:
                    recommendations.append(f"📊 **Disconnected network** - {n_components} separate variable clusters detected")
                    recommendations.append("💡 Consider analyzing clusters independently or finding bridging variables")
            
            # Clustering insights
            clustering = best_topology.get("average_clustering", 0)
            if clustering > 0.5:
                recommendations.append("🔘 **High clustering** - variables form tight-knit groups")
            elif clustering > 0.3:
                recommendations.append("🔗 **Moderate clustering** - some variable groupings detected")
            else:
                recommendations.append("📈 **Low clustering** - variables have sparse local connections")
            
            # Small world insights
            if best_topology.get("is_small_world", False):
                recommendations.append("🌍 **Small-world network** - efficient global connectivity with local clustering")
                recommendations.append("🚀 Ideal for feature engineering and dimensionality reduction")
            
            # Community insights
            community_methods = [method for method in best_communities.keys() if "error" not in best_communities[method]]
            if community_methods:
                # Find best community detection
                best_community_method = None
                best_modularity = -1
                
                for method in community_methods:
                    comm_info = best_communities[method]
                    modularity = comm_info.get("modularity", -1)
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_community_method = method
                
                if best_community_method and best_modularity > 0.3:
                    n_communities = best_communities[best_community_method]["n_communities"]
                    recommendations.append(f"🏘️ **{n_communities} communities detected** using {best_community_method.replace('_', ' ').title()}")
                    recommendations.append(f"📈 High modularity ({best_modularity:.3f}) indicates strong community structure")
                elif best_modularity > 0:
                    recommendations.append("🔍 **Weak community structure** - variables are somewhat grouped")
                else:
                    recommendations.append("🌐 **No clear communities** - variables are uniformly connected")
            
            # Centrality insights
            if "degree" in best_centralities and "error" not in best_centralities["degree"]:
                top_degree_nodes = best_centralities["degree"]["top_nodes"]
                if top_degree_nodes:
                    top_node = top_degree_nodes[0][0]
                    recommendations.append(f"⭐ **Most connected variable**: {top_node}")
            
            if "pagerank" in best_centralities and "error" not in best_centralities["pagerank"]:
                top_pagerank_nodes = best_centralities["pagerank"]["top_nodes"]
                if top_pagerank_nodes:
                    top_node = top_pagerank_nodes[0][0]
                    recommendations.append(f"👑 **Most influential variable**: {top_node}")
            
            # Embedding insights
            embeddings = results["embeddings"].get(best_graph, {})
            if embeddings and "node2vec" in embeddings and "error" not in embeddings["node2vec"]:
                recommendations.append("🧠 **Node embeddings available** - use for advanced ML feature engineering")
                
                similar_pairs = embeddings["node2vec"].get("most_similar_pairs", [])
                if similar_pairs:
                    most_similar = similar_pairs[0]
                    recommendations.append(f"🔗 **Most similar variables**: {most_similar[0]} ↔ {most_similar[1]} (similarity: {most_similar[2]:.3f})")
        
        # Data-specific recommendations
        data_info = results["data_info"]
        
        if data_info["was_subsampled"]:
            recommendations.append("📊 **Large dataset subsampled** - results are representative but may miss rare patterns")
        
        if len(data_info["constant_columns_removed"]) > 0:
            recommendations.append(f"🚫 **{len(data_info['constant_columns_removed'])} constant columns removed** - no network information")
        
        # Graph construction recommendations
        successful_graphs = results["summary"]["successful_constructions"]
        total_attempted = results["summary"]["total_attempted"]
        
        if successful_graphs == total_attempted:
            recommendations.append("✅ **All graph construction methods successful** - robust network structure")
        elif successful_graphs > 0:
            recommendations.append(f"⚠️ **{successful_graphs}/{total_attempted} graph methods successful** - some approaches failed")
        else:
            recommendations.append("❌ **No successful graph constructions** - data may lack network structure")
        
        # Advanced analysis recommendations
        if best_graph:
            best_nodes = results["summary"].get("best_graph_nodes", 0)
            best_edges = results["summary"].get("best_graph_edges", 0)
            
            if best_nodes >= 10 and best_edges >= 20:
                recommendations.append("🔬 **Rich network structure** - suitable for advanced graph ML algorithms")
            
            if best_topology.get("density", 0) > 0.5:
                recommendations.append("🕸️ **Dense network** - consider edge pruning or sparsification")
            elif best_topology.get("density", 0) < 0.1:
                recommendations.append("🌿 **Sparse network** - may benefit from relaxed connection thresholds")
        
        results["recommendations"] = recommendations[:8]  # Limit to top recommendations

    def get_network_visualization_data(self, results: Dict[str, Any], graph_type: str = None) -> Dict[str, Any]:
        """
        Extract data needed for network visualization
        
        Args:
            results: Results from analyze() method
            graph_type: Specific graph type to visualize (if None, uses best)
            
        Returns:
            Dictionary with nodes, edges, and layout information for visualization
        """
        if graph_type is None:
            graph_type = results["summary"].get("best_graph_type")
        
        if not graph_type or graph_type not in results["graphs"]:
            return {"error": "No suitable graph found for visualization"}
        
        graph_info = results["graphs"][graph_type]
        if "error" in graph_info:
            return {"error": f"Graph construction failed: {graph_info['error']}"}
        
        # Get the NetworkX graph object
        G = graph_info.get("graph_object")
        if G is None:
            return {"error": "Graph object not available"}
        
        # Extract node information
        nodes = []
        centralities = results["centralities"].get(graph_type, {})
        communities = results["communities"].get(graph_type, {})
        
        # Get node sizes from degree centrality
        degree_cent = centralities.get("degree", {}).get("values", {})
        pagerank_cent = centralities.get("pagerank", {}).get("values", {})
        
        # Get community assignments
        community_assignment = {}
        if communities:
            for method in ["louvain", "spectral", "label_propagation"]:
                if method in communities and "partition" in communities[method]:
                    community_assignment = communities[method]["partition"]
                    break
        
        for node in G.nodes():
            node_info = {
                "id": node,
                "label": node,
                "size": degree_cent.get(node, 0.1) * 100 + 10,  # Scale for visualization
                "degree": G.degree(node),
                "pagerank": pagerank_cent.get(node, 0),
                "community": community_assignment.get(node, 0)
            }
            nodes.append(node_info)
        
        # Extract edge information
        edges = []
        for u, v, data in G.edges(data=True):
            edge_info = {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "type": data.get("type", "unknown")
            }
            edges.append(edge_info)
        
        # Compute layout positions using spring layout
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
            for node_info in nodes:
                node_id = node_info["id"]
                if node_id in pos:
                    node_info["x"] = float(pos[node_id][0])
                    node_info["y"] = float(pos[node_id][1])
        except:
            # Fallback to random positions
            np.random.seed(42)
            for i, node_info in enumerate(nodes):
                node_info["x"] = np.random.uniform(-1, 1)
                node_info["y"] = np.random.uniform(-1, 1)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "graph_type": graph_type,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "layout": "spring",
            "communities": len(set(community_assignment.values())) if community_assignment else 1
        }
