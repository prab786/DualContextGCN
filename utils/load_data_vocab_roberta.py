import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import spacy
from transformers import RobertaTokenizer, RobertaModel
import os
import math
from collections import defaultdict, Counter
from itertools import combinations
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def tokenize_roberta(texts, tokenizer):
    """Tokenize texts using RoBERTa tokenizer"""
    tokenized_docs = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        tokenized_docs.append(tokens)
    return tokenized_docs


def build_vocab_and_cooc(tokenized_docs, window_size=10, min_count=0):
    """Build vocabulary and co-occurrence statistics"""
    vocab_counter = Counter()
    co_occurrence = defaultdict(int)
    total_windows = 0

    for tokens in tokenized_docs:
        for i in range(len(tokens)):
            window = tokens[i:i+window_size]
            unique = set(window)
            total_windows += 1

            for token in unique:
                vocab_counter[token] += 1
            for t1, t2 in combinations(sorted(unique), 2):
                co_occurrence[(t1, t2)] += 1

    vocab = {w for w, c in vocab_counter.items() if c >= min_count}
    token2idx = {token: idx for idx, token in enumerate(sorted(vocab))}

    return vocab, token2idx, vocab_counter, co_occurrence, total_windows


def npmi(w1, w2, vocab_counter, co_occurrence, total_windows):
    """Calculate Normalized Pointwise Mutual Information"""
    p_x = vocab_counter[w1] / total_windows
    p_y = vocab_counter[w2] / total_windows
    p_xy = co_occurrence[(w1, w2)] / total_windows
    if p_xy == 0:
        return None
    pmi = math.log(p_xy / (p_x * p_y) + 1e-10)
    return pmi / (-math.log(p_xy + 1e-10))


def compute_npmi_edges(token2idx, vocab_counter, co_occurrence, total_windows):
    """Compute edges based on NPMI scores"""
    edges = []
    weights = []

    for (w1, w2), _ in co_occurrence.items():
        if w1 in token2idx and w2 in token2idx:
            score = npmi(w1, w2, vocab_counter, co_occurrence, total_windows)
            if score and score > 0:
                edges.append((token2idx[w1], token2idx[w2]))
                weights.append(score)

    return edges, weights


def build_pyg_graph_efficient(texts, tokenizer=tokenizer, window_size=10, min_count=1, 
                            node_feature_dim=768, use_roberta_init=False):
    """Efficient graph building with proper node feature dimensions"""
    tokenized = tokenize_roberta(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    num_tokens = len(token2idx)
    print(f"Building efficient graph: {num_tokens} tokens, {len(edges)} edges")
    print(f"Node feature dimension: {node_feature_dim}")
    
    if not edges:
        print("Warning: No edges found, creating graph with isolated nodes")
        x = torch.randn(max(1, num_tokens), node_feature_dim) * 0.1
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # Create edge tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Create appropriately sized node features
    if use_roberta_init:
        print("Initializing with RoBERTa embeddings (this may take a while)...")
        x = initialize_with_roberta_embeddings(token2idx, tokenizer, node_feature_dim)
    else:
        # Simple random initialization (much faster)
        x = torch.randn(num_tokens, node_feature_dim) * 0.1
        print(f"Initialized with random features: {x.shape}")

    # Verify everything is consistent
    assert x.size(0) == num_tokens, f"Node features size {x.size(0)} != vocab size {num_tokens}"
    assert edge_index.max().item() < num_tokens, f"Edge index {edge_index.max().item()} >= vocab size {num_tokens}"
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx


def initialize_with_roberta_embeddings(token2idx, tokenizer, embedding_dim=768):
    """Initialize node features with RoBERTa embeddings"""
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    roberta_model.eval()
    
    num_tokens = len(token2idx)
    x = torch.zeros(num_tokens, embedding_dim)
    
    # Process tokens in batches for efficiency
    tokens = list(token2idx.keys())
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            
            for token in batch_tokens:
                try:
                    # Encode single token
                    token_ids = tokenizer.encode(token, add_special_tokens=True, return_tensors='pt')
                    outputs = roberta_model(token_ids)
                    
                    # Use average of all token embeddings
                    token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = token_embedding[:embedding_dim]  # Ensure correct size
                    
                except Exception as e:
                    print(f"Warning: Could not get RoBERTa embedding for token '{token}': {e}")
                    # Use random initialization for problematic tokens
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = torch.randn(embedding_dim) * 0.1
    
    return x


def get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer):
    """Robust version of get_vocab_subgraph_from_text with better error handling"""
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        if not token2idx:
            raise ValueError("Empty token2idx vocabulary")
        
        # Tokenize the input text
        tokens = tokenizer.tokenize(text.lower())
        
        if not tokens:
            raise ValueError("No tokens found in text")
        
        # Map to graph node indices - remove duplicates while preserving order
        seen = set()
        node_indices = []
        for token in tokens:
            if token in token2idx and token not in seen:
                node_indices.append(token2idx[token])
                seen.add(token)
        
        if not node_indices:
            raise ValueError("None of the tokens from the input text are in the vocabulary graph")

        node_mask = torch.tensor(node_indices, dtype=torch.long)

        # Extract the subgraph
        edge_index, edge_attr = subgraph(
            node_mask, 
            vocab_graph.edge_index, 
            edge_attr=vocab_graph.edge_attr, 
            relabel_nodes=True
        )

        # Get node features
        x = vocab_graph.x[node_mask]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    except Exception as e:
        print(f"Warning: Error in subgraph extraction: {e}")
        # Return minimal valid subgraph
        device = vocab_graph.x.device if hasattr(vocab_graph, 'x') and vocab_graph.x is not None else torch.device('cpu')
        return Data(
            x=torch.zeros(1, 768).to(device),
            edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
            edge_attr=torch.zeros(0).to(device)
        )


def split_news_by_binary_label(train_dict):
    """Split news texts into two lists based on binary labels (0 or 1)"""
    news = train_dict['news']
    labels = train_dict['labels']

    if len(news) != len(labels):
        raise ValueError("The number of news items must match the number of labels.")

    news_0 = [n for n, l in zip(news, labels) if l == 0]
    news_1 = [n for n, l in zip(news, labels) if l == 1]

    return news_0, news_1


def load_vocab_graphs(file_path, texts, force_rebuild=False):
    """Load or build vocabulary graphs with correct RoBERTa-sized features"""
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            # Validate the data structure
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                
                # Check node feature dimensions
                if graph.x is not None:
                    node_feature_dim = graph.x.size(1)
                    num_nodes = graph.x.size(0)
                    vocab_size = len(token2idx)
                    
                    print(f"Loaded graph: {num_nodes} nodes, {vocab_size} vocab size, {node_feature_dim}D features")
                    
                    # Check if we have the old problematic format
                    if node_feature_dim == vocab_size:
                        print("⚠️ Detected old identity matrix format! Forcing rebuild...")
                        force_rebuild = True
                    elif node_feature_dim != 768:
                        print(f"⚠️ Non-RoBERTa feature dimension ({node_feature_dim})! Forcing rebuild...")
                        force_rebuild = True
                    else:
                        print("✅ Graph format is compatible")
                        return vocab_data
                else:
                    print("⚠️ Graph has no node features! Forcing rebuild...")
                    force_rebuild = True
            else:
                print("⚠️ Invalid graph format! Forcing rebuild...")
                force_rebuild = True
                
        except Exception as e:
            print(f"Error loading graph: {e}. Forcing rebuild...")
            force_rebuild = True
    
    if force_rebuild or not file_exists(file_path):
        print(f"Building new vocabulary graph for {len(texts)} texts...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build the graph with correct dimensions
        graph, token2idx = build_pyg_graph_efficient(
            texts, 
            tokenizer=tokenizer, 
            window_size=10, 
            min_count=1,
            node_feature_dim=768,  # RoBERTa dimension
            use_roberta_init=False
        )
        
        # Save the new graph
        vocab_graph = {
            0: graph,
            1: token2idx,
            'metadata': {
                'num_texts': len(texts),
                'num_tokens': len(token2idx),
                'node_feature_dim': 768,
                'num_edges': graph.edge_index.size(1) if graph.edge_index is not None else 0
            }
        }
        
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"✅ Built and saved new graph: {len(token2idx)} tokens, 768D features")
        return vocab_graph


def file_exists(file_path):
    """Check if file exists"""
    return os.path.exists(file_path)


def load_articles(obj):
    """Load news articles for training and testing"""
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))   

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    

    return x_train, x_test,  y_train, y_test