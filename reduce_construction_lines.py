import sys
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataloader import cad2sketch_dataset_loader
from torch.utils.data import DataLoader

import gnn.gan
import helper
import gnn_graph
import gnn.gnn
import os

import numpy as np

import cad2sketch_stroke_features



current_dir = os.getcwd()
save_dir = os.path.join(current_dir, 'checkpoints', 'reduce_construction')
os.makedirs(save_dir, exist_ok=True)

def load_models():
    graph_encoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_encoder.pth')))
    graph_decoder.load_state_dict(torch.load(os.path.join(save_dir, 'graph_decoder.pth')))


def save_models():
    torch.save(graph_encoder.state_dict(), os.path.join(save_dir, 'graph_encoder.pth'))
    torch.save(graph_decoder.state_dict(), os.path.join(save_dir, 'graph_decoder.pth'))


# ------------------------------------------------------------------------------# 


# Initialize graph encoder and decoder for a gnn
# graph_encoder = gnn.gnn.SemanticModule()
# graph_decoder = gnn.gnn.Stroke_Decoder()

#Initialize graph encoder and decoder for a gan
graph_encoder = gnn.gan.SemanticGANModule()
graph_decoder = gnn.gan.Stroke_Decoder_GAN()


# Move models to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph_encoder.to(device)
graph_decoder.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(list(graph_encoder.parameters()) + list(graph_decoder.parameters()), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss



def train():
    # Initialize dataset
    dataset = cad2sketch_dataset_loader()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    graphs = []
    final_edges_mask = []

    # Load data
    for data in tqdm(dataloader, desc="Building Graphs"):
        intersection_matrix, all_edges_matrix, final_edges_matrix, all_edges_file_path= data
        
        all_edges_matrix = all_edges_matrix.squeeze(0)[:, :6].to(device)  # Shape: (num_strokes, 7)
        intersection_matrix = intersection_matrix.squeeze(0).to(device)  # Shape: (num_strokes, num_strokes)
        final_edges_matrix = final_edges_matrix.squeeze(0).to(device)  # Shape: (num_strokes, 1)

        cur_gnn_graph = gnn_graph.StrokeGraph(all_edges_matrix, intersection_matrix)

        graphs.append(cur_gnn_graph)
        final_edges_mask.append(final_edges_matrix)


    # Split dataset
    split_index = int(0.8 * len(graphs))
    train_graphs, val_graphs = graphs[:split_index], graphs[split_index:]
    train_masks, val_masks = final_edges_mask[:split_index], final_edges_mask[split_index:]

    # Training loop
    epochs = 30
    best_accuracy = 0.0

    for epoch in range(epochs):
        graph_encoder.train()
        graph_decoder.train()
        total_loss = 0.0

        # Training loop
        for graph, mask in tqdm(zip(train_graphs, train_masks)):
            optimizer.zero_grad()

            # Forward pass
            x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
            output = graph_decoder(x_dict)  # Shape: (num_strokes, 1)

            # Compute loss
            loss = criterion(output, mask)  # BCEWithLogitsLoss expects (batch, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_graphs)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation loop
        graph_encoder.eval()
        graph_decoder.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for graph, mask in tqdm(zip(val_graphs, val_masks)):
                x_dict = graph_encoder(graph.x_dict, graph.edge_index_dict)
                output = graph_decoder(x_dict)  # Shape: (num_strokes, 1)

                # Convert logits to probabilities (sigmoid activation)
                probs = torch.sigmoid(output)

                # Convert to binary predictions (threshold at 0.5)
                pred = (probs > 0.5).float()

                # Compute accuracy
                correct += (pred == mask).sum().item()
                total += mask.numel()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_models()
            print("Saved new best model.")

    print("Training complete. Best validation accuracy:", best_accuracy)




# ------------------------------------------------------------------------------# 


def eval():
    """Evaluate the trained model and visualize prediction results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained models
    load_models()

    # Load dataset
    dataset = cad2sketch_dataset_loader()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in dataloader:
        intersection_matrix, all_edges_matrix, final_edges_matrix, all_edges_file_path = data
        
        all_edges_matrix = all_edges_matrix.squeeze(0)[:, :6].to(device)  # Shape: (num_strokes, 7)
        intersection_matrix = intersection_matrix.squeeze(0).to(device)  # Shape: (num_strokes, num_strokes)
        final_edges_matrix = final_edges_matrix.squeeze(0).to(device)  # Shape: (num_strokes, 1)

        # Create graph
        cur_gnn_graph = gnn_graph.StrokeGraph(all_edges_matrix, intersection_matrix)

        # Forward pass
        x_dict = graph_encoder(cur_gnn_graph.x_dict, cur_gnn_graph.edge_index_dict)
        output = graph_decoder(x_dict)  # Shape: (num_strokes, 1)

        # Convert to binary predictions
        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.5).float()  # Predicted stroke selection

        # Convert tensors back to CPU
        pred_mask = pred_mask.cpu().numpy()
        final_edges_matrix = final_edges_matrix.cpu().numpy()

        # Visualize results
        all_edges_data = helper.read_json(all_edges_file_path[0])
        cad2sketch_stroke_features.vis_all_edges_selected(all_edges_data,pred_mask)
        cad2sketch_stroke_features.vis_all_edges_selected(all_edges_data,final_edges_matrix)

        cad2sketch_stroke_features.vis_all_edges_only_selected(all_edges_data,pred_mask)
        cad2sketch_stroke_features.vis_all_edges_only_selected(all_edges_data,final_edges_matrix)



# ------------------------------------------------------------------------------# 
train()

