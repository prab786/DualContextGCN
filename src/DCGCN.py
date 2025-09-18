# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
import sys, os
import pickle
sys.path.append(os.getcwd())
from utils.load_data_vocab_roberta import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
warnings.filterwarnings("ignore")

# Command line arguments
parser = argparse.ArgumentParser(description='Dual Vocabulary Graph Model for Fake News Detection')
parser.add_argument('--dataset_name', default='politifact', type=str, help='Dataset name: politifact/gossipcop')
parser.add_argument('--model_name', default='DCGCN-', type=str, help='Model name for logging')
parser.add_argument('--iters', default=10, type=int, help='Number of iterations')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--n_epochs', default=3, type=int, help='Number of epochs')
parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--graph_hidden_dim', default=768, type=int, help='Graph hidden dimension')
parser.add_argument('--fusion_dim', default=156, type=int, help='Fusion layer dimension')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)


class DualVocabNewsDataset(Dataset):
    """Dataset for dual vocabulary graph model with dynamic node features"""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # RoBERTa tokenization
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length',
            truncation=True, 
            return_token_type_ids=False, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        return {
            'original_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


class WeightedAttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which nodes are most important.
    Think of this as having a smart dispatcher who knows which vocabulary 
    terms are most indicative of fake news patterns.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(WeightedAttentionPooling, self).__init__()
        
        # The attention network learns to identify important nodes
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Single attention score per node
            nn.Sigmoid()  # Attention weights between 0 and 1
        )
        
        # Optional: Add position encoding for vocabulary term importance
        self.use_position_encoding = False
        if self.use_position_encoding:
            self.position_encoder = nn.Linear(1, input_dim // 4)
    
    def forward(self, x, batch=None, vocab_term_frequencies=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment for each node (for batched processing)
            vocab_term_frequencies: Optional frequency information for each term
        """
        if x.size(0) == 0:
            return torch.zeros(x.size(1)).to(x.device)
        
        # Add position encoding if vocabulary frequency information is available
        if self.use_position_encoding and vocab_term_frequencies is not None:
            freq_encoding = self.position_encoder(vocab_term_frequencies.unsqueeze(-1))
            x_with_position = torch.cat([x, freq_encoding], dim=-1)
        else:
            x_with_position = x
        
        # Compute attention weights for each node
        attention_weights = self.attention_net(x_with_position)  # [num_nodes, 1]
        
        # Apply attention weights to node features
        weighted_features = x * attention_weights  # [num_nodes, input_dim]
        
        # Global pooling with attention weights
        if batch is None:
            # Single graph case
            pooled = torch.sum(weighted_features, dim=0)  # [input_dim]
            # Normalize by sum of attention weights to maintain scale
            total_attention = torch.sum(attention_weights) + 1e-8
            pooled = pooled / total_attention
        else:
            # Batched graphs case - use scatter operations
            from torch_scatter import scatter_add
            pooled = scatter_add(weighted_features, batch, dim=0)
            attention_sums = scatter_add(attention_weights, batch, dim=0) + 1e-8
            pooled = pooled / attention_sums.unsqueeze(-1)
        #print(f"Attention weights shape: {attention_weights.shape}, Pooled shape: {pooled.shape}")
        return pooled.squeeze(0).squeeze(0), attention_weights.squeeze(-1)  # Return weights for explainability



class GeGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SimplifiedDualVocabModel(nn.Module):
    """Simplified dual vocabulary model with configurable RoBERTa layer node features"""
    
    def __init__(self, roberta_dim=768, graph_hidden_dim=768, fusion_dim=156, num_classes=2, dropout=0.3, target_layer=9):
        super(SimplifiedDualVocabModel, self).__init__()
        
        self.roberta_dim = roberta_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        self.target_layer = target_layer
        
        # RoBERTa encoder - enable output of all hidden states
        self.roberta = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
        self.roberta_dropout = nn.Dropout(dropout)
       
        self.real_graph_conv1 = GCNConv(roberta_dim, graph_hidden_dim)
        self.real_graph_conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
        self.fake_graph_conv1 = GCNConv(roberta_dim, graph_hidden_dim)
        self.fake_graph_conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)        
        self.graph_dropout = nn.Dropout(dropout)  
        self.graph_pooling = WeightedAttentionPooling(graph_hidden_dim)  
        # Normalization
        self.roberta_norm = nn.LayerNorm(roberta_dim)
        self.graph_norm = nn.LayerNorm(graph_hidden_dim)        
        self.classifer = nn.Linear(2*fusion_dim, num_classes)
        self.Linear_layer_f = nn.Linear(graph_hidden_dim, fusion_dim)
        self.Linear_layer_r = nn.Linear(graph_hidden_dim, fusion_dim)
        
      
    
    def forward(self, input_ids, attention_mask, original_texts, real_vocab_graph, 
                fake_vocab_graph, real_token2idx, fake_token2idx, tokenizer):
        batch_size = input_ids.size(0)
        
        # RoBERTa processing with all hidden states
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract target layer embeddings
        target_layer_embeddings = roberta_outputs.last_hidden_state
        roberta_pooled = roberta_outputs.pooler_output
        
        # Process each sample in the batch with target layer embeddings
        real_graph_features_batch = []
        fake_graph_features_batch = []
        
        for i in range(batch_size):
            sample_text = original_texts[i]
            sample_target_embeddings = target_layer_embeddings[i]
            
            # Get contextualized subgraphs using target layer embeddings
            real_subgraph = self._get_contextualized_subgraph(
                sample_text, real_vocab_graph, real_token2idx, 
                sample_target_embeddings, tokenizer
            )
            fake_subgraph = self._get_contextualized_subgraph(
                sample_text, fake_vocab_graph, fake_token2idx,
                sample_target_embeddings, tokenizer
            )
            
            # Process graphs
            real_features = self._process_graph(real_subgraph, self.real_graph_conv1, self.real_graph_conv2)
            fake_features = self._process_graph(fake_subgraph, self.fake_graph_conv1, self.fake_graph_conv2)
            
            real_graph_features_batch.append(real_features)
            fake_graph_features_batch.append(fake_features)
        
        # Stack batch results
        real_graph_features = torch.stack(real_graph_features_batch) #H_Real
        fake_graph_features = torch.stack(fake_graph_features_batch) #H_Fake
        
        # Generate predictions
        h_fake = self.Linear_layer_f(fake_graph_features)
        h_real = self.Linear_layer_f(real_graph_features)
        
        fused_features_logit = torch.cat([h_real, h_fake], dim=-1)
        logits = self.classifer(fused_features_logit)
        
        return {
            'logits': logits,            
            'roberta_features': roberta_pooled,
            'real_graph_features': h_real,
            'fake_graph_features': h_fake,           
                      
        }
    
    def _get_contextualized_subgraph(self, input_text, vocab_graph, token2idx, target_layer_embeddings, tokenizer):
        """Get subgraph with contextualized RoBERTa target layer node features"""
        try:
            base_subgraph = get_vocab_subgraph_from_text_robust(input_text, vocab_graph, token2idx, tokenizer)
            updated_subgraph = self._update_node_features_with_roberta(
                base_subgraph, input_text, target_layer_embeddings, token2idx, tokenizer
            )
            return updated_subgraph
            
        except Exception as e:
            print(f"Warning: Error in contextualized subgraph creation: {e}")
            device = target_layer_embeddings.device if hasattr(target_layer_embeddings, 'device') else torch.device('cpu')
            return Data(
                x=torch.zeros(1, 768).to(device),
                edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
                edge_attr=torch.zeros(0).to(device)
            )
            
    def _update_node_features_with_roberta(self, subgraph, input_text, target_layer_embeddings, token2idx, tokenizer):
        """Update subgraph node features with RoBERTa target layer embeddings"""
        if subgraph.x is None or subgraph.x.size(0) == 0:
            return subgraph
        
        # Get RoBERTa tokens
        roberta_tokens = tokenizer.tokenize(input_text.lower())
        
        # Create mapping from vocabulary tokens to RoBERTa positions
        roberta_token_positions = {}
        for i, roberta_token in enumerate(roberta_tokens):
            if roberta_token in token2idx:
                roberta_position = i + 1  # +1 for <s> token
                if roberta_position < target_layer_embeddings.size(0):
                    roberta_token_positions[roberta_token] = roberta_position
        
        # Get tokens present in both input text and vocabulary
        text_tokens_in_vocab = []
        for token in roberta_tokens:
            if token in token2idx:
                text_tokens_in_vocab.append(token)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in text_tokens_in_vocab:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        original_node_indices = [token2idx[token] for token in unique_tokens]
        token_index_pairs = list(zip(unique_tokens, original_node_indices))
        token_index_pairs.sort(key=lambda x: x[1])
        
        # Update node features
        updated_node_features = subgraph.x.clone()
        
        for subgraph_idx, (token, original_vocab_idx) in enumerate(token_index_pairs):
            if subgraph_idx >= updated_node_features.size(0):
                break
                
            if token in roberta_token_positions:
                roberta_pos = roberta_token_positions[token]
                updated_node_features[subgraph_idx] = target_layer_embeddings[roberta_pos]
        
        return Data(
            x=updated_node_features.to(target_layer_embeddings.device),
            edge_index=subgraph.edge_index.to(target_layer_embeddings.device),
            edge_attr=subgraph.edge_attr.to(target_layer_embeddings.device)
        )
    
    def _process_graph(self, subgraph, conv1, conv2):
        """Process a single subgraph"""
        if subgraph.x is None or subgraph.x.size(0) == 0:
            return torch.zeros(self.graph_hidden_dim).to(next(self.parameters()).device)
        
        device = next(self.parameters()).device
        x = subgraph.x.to(device)
        edge_index = subgraph.edge_index.to(device)
        
        # Apply graph convolutions
        if edge_index.size(1) > 0:
            x = F.relu(conv1(x, edge_index))
            x = F.relu(conv2(x, edge_index))
        
        # Pooling
        if x.size(0) == 1:
            pooled = x.squeeze(0)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
            pooled, _ = self.graph_pooling(x, batch)
        
        return self.graph_norm(pooled)


class DualVocabClassifier(nn.Module):
    """Dual vocabulary classifier with configurable RoBERTa layer node features"""
    
    def __init__(self, real_vocab_size, fake_vocab_size, n_classes=2):
        super(DualVocabClassifier, self).__init__()
        
        self.dual_vocab_model = SimplifiedDualVocabModel(
            roberta_dim=768,
            graph_hidden_dim=args.graph_hidden_dim,
            fusion_dim=args.fusion_dim,
            num_classes=n_classes,
            dropout=args.dropout,
           
        )

    def forward(self, input_ids, attention_mask, original_texts, real_vocab_graph, 
                fake_vocab_graph, real_token2idx, fake_token2idx, tokenizer):
        outputs = self.dual_vocab_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_texts=original_texts,
            real_vocab_graph=real_vocab_graph,
            fake_vocab_graph=fake_vocab_graph,
            real_token2idx=real_token2idx,
            fake_token2idx=fake_token2idx,
            tokenizer=tokenizer
        )
        
        return outputs['logits'], outputs


def collate_fn(batch):
    """Custom collate function for dynamic node feature processing"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    original_texts = [item['original_text'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_texts': original_texts
    }


def create_data_loader(texts, labels, tokenizer, max_len, batch_size, shuffle=False):
    """Create data loader for dynamic dual vocabulary model"""
    ds = DualVocabNewsDataset(
        texts=texts, labels=np.array(labels), tokenizer=tokenizer, max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter_num,
               real_vocab_graph, fake_vocab_graph, real_token2idx, fake_token2idx):
    """Training function for dual vocabulary graph model with configurable RoBERTa layer features"""
    
    print(f"Loading data for {datasetname}...")
    
    try:
        load_result = load_articles(datasetname)
        print(f"load_articles returned {len(load_result)} values")        
       
        x_train, x_test, y_train, y_test = load_result[:5]
       
            
    except Exception as e:
        print(f"Error loading articles: {e}")
        raise e

    print(f"Loaded {len(x_train)} training samples, {len(x_test)} test samples")
    
    # Create data loaders
    train_loader = create_data_loader(
        x_train, y_train, tokenizer, max_len, batch_size, shuffle=True
    )
    test_loader = create_data_loader(
        x_test, y_test, tokenizer, max_len, batch_size, shuffle=False
    )
    

    # Initialize model
    model = DualVocabClassifier(
        real_vocab_size=len(real_token2idx),
        fake_vocab_size=len(fake_token2idx),
        n_classes=2
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = n_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
    )

    train_losses = []
    train_accs = []

    for epoch in range(n_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{n_epochs} - Training with RoBERTa layer  embeddings...")

        avg_loss = []
        avg_acc = []

        for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{n_epochs}"):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            targets = batch_data["labels"].to(device)
            original_texts = batch_data["original_texts"]
            
            try:
                binary_output, output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    original_texts=original_texts,
                    real_vocab_graph=real_vocab_graph,
                    fake_vocab_graph=fake_vocab_graph,
                    real_token2idx=real_token2idx,
                    fake_token2idx=fake_token2idx,
                    tokenizer=tokenizer
                )
                
                # Compute loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(binary_output, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                avg_loss.append(loss.item())
                
                # Calculate accuracy
                _, pred = binary_output.max(dim=-1)
                correct = pred.eq(targets).sum().item()
                train_acc = correct / len(targets)
                avg_acc.append(train_acc)
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        train_losses.append(np.mean(avg_loss) if avg_loss else 0.0)
        train_accs.append(np.mean(avg_acc) if avg_acc else 0.0)

        current_acc = np.mean(avg_acc) if avg_acc else 0.0
        current_loss = np.mean(avg_loss) if avg_loss else 0.0
        print(f"Iter {iter_num:03d} | Epoch {epoch:05d} | Train Acc. {current_acc:.4f} | Loss {current_loss:.4f}")

        # Evaluation at the end of training
        if epoch == n_epochs - 1:
            model.eval()
            y_pred = []
            y_pred_res = []
            y_true = []

            # Evaluate on original test set
            for batch_data in tqdm(test_loader, desc="Evaluating original test set"):
                with torch.no_grad():
                    input_ids = batch_data["input_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)
                    targets = batch_data["labels"].to(device)
                    original_texts = batch_data["original_texts"]
                    
                    val_out, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        original_texts=original_texts,
                        real_vocab_graph=real_vocab_graph,
                        fake_vocab_graph=fake_vocab_graph,
                        real_token2idx=real_token2idx,
                        fake_token2idx=fake_token2idx,
                        tokenizer=tokenizer
                    )
                    _, val_pred = val_out.max(dim=1)

                    y_pred.append(val_pred)
                    y_true.append(targets)

            
            # Compute metrics
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)           

            acc = accuracy_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')

          

    # Save model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{datasetname}_dual_vocab__iter{iter_num}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),        
        'args': args,
        'epoch': n_epochs,
        'accuracy': acc
    }, checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")

    print(f"-----------------End of Iter {iter_num:03d}-----------------")
    print([f'Global Test Accuracy:{acc:.4f}',
           f'Precision:{precision:.4f}',
           f'Recall:{recall:.4f}',
           f'F1:{fscore:.4f}'])

  

    return acc, precision, recall, fscore


def main():
    """Main training function"""
    try:
        datasetname = args.dataset_name
        batch_size = args.batch_size
        max_len = args.max_len
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        n_epochs = args.n_epochs
        iterations = args.iters

        print(f"Starting training with configuration:")
        print(f"  Dataset: {datasetname}")
       
        print(f"  Batch Size: {batch_size}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Iterations: {iterations}")

        # Load vocabulary graphs for the dataset
        print("Loading vocabulary graphs...")
        train_dict = pickle.load(open(f'data/news_articles/{datasetname}_train.pkl', 'rb'))
        news_real, news_fake = split_news_by_binary_label(train_dict)
        
        # Load pre-built vocabulary graphs 
        x_train_graph_real = load_vocab_graphs(f'gr/vocab_r_{datasetname}_roberta.Rograph', news_real)
        x_train_graph_fake = load_vocab_graphs(f'gr/vocab_f_{datasetname}_roberta.Rograph', news_fake)
        
        real_vocab_graph = x_train_graph_real[0]
        real_token2idx = x_train_graph_real[1]
        fake_vocab_graph = x_train_graph_fake[0]
        fake_token2idx = x_train_graph_fake[1]

        print(f"Real vocabulary size: {len(real_token2idx)}")
        print(f"Fake vocabulary size: {len(fake_token2idx)}")

        # Initialize result tracking
        test_accs = []
        prec_all, rec_all, f1_all = [], [], []
        test_accs_res = []
        prec_all_res, rec_all_res, f1_all_res = [], [], []

        # Run multiple iterations with different seeds for robust evaluation
        for iter_num in range(iterations):
            print(f"\n{'='*50}")
            print(f"Starting iteration {iter_num + 1}/{iterations}")
            print(f"{'='*50}")
            
            set_seed(iter_num)
            
            acc, prec, recall, f1= train_model(
                tokenizer, max_len, n_epochs, batch_size, datasetname, iter_num,
                real_vocab_graph, fake_vocab_graph, real_token2idx, fake_token2idx
            )

            test_accs.append(acc)
            prec_all.append(prec)
            rec_all.append(recall)
            f1_all.append(f1)
           

        # Print final results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        print("Original Test Set:")
        print(f"Total_Test_Accuracy: {sum(test_accs) / iterations:.4f} | "
              f"Prec_Macro: {sum(prec_all) / iterations:.4f} | "
              f"Rec_Macro: {sum(rec_all) / iterations:.4f} | "
              f"F1_Macro: {sum(f1_all) / iterations:.4f}")

      

        # Save results to log file
        os.makedirs('logs', exist_ok=True)
        log_filename = f'logs/log_{datasetname}_{args.model_name}__iter{iterations}.txt'
        
        with open(log_filename, 'w') as f:
            f.write('-------------Dual Vocabulary Graph Model Results (RoBERTa)-------------\n')
            f.write(f'Dataset: {datasetname}\n')
            f.write(f'Model: {args.model_name}\n')            
            f.write(f'Iterations: {iterations}\n')
            f.write(f'Epochs: {n_epochs}\n')
            f.write(f'Batch Size: {batch_size}\n')
            f.write(f'Learning Rate: {args.learning_rate}\n')
           
            
            f.write('-------------Original Test Set-------------\n')
            f.write(f'All Acc.s: {test_accs}\n')
            f.write(f'All Prec.s: {prec_all}\n')
            f.write(f'All Rec.s: {rec_all}\n')
            f.write(f'All F1.s: {f1_all}\n')
            f.write(f'Average acc.: {sum(test_accs) / iterations}\n')
            f.write(f'Average Prec / Rec / F1 (macro): {sum(prec_all) / iterations}, {sum(rec_all) / iterations}, {sum(f1_all) / iterations}\n')
          
        print(f"\nResults saved to: {log_filename}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
