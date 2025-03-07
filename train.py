import argparse
import numpy as np
import wandb
from dataset import load_data
from model import FeedForwardNN
from optimizer import get_optimizer
from loss import get_loss
from utils import accuracy
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(description='Train a feedforward neural network')
    
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname')#DA6401-A1
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname')#da24m013-iit-madras-alumni-association
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-beta', '--beta', type=float, default=0.9)
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9)
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-3)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0)
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier', choices=['random', 'Xavier'])
    parser.add_argument('-nhl', '--num_layers', type=int, default=5)
    parser.add_argument('-sz', '--hidden_size', type=int, default=128)
    parser.add_argument('-a', '--activation', type=str, default='ReLU', choices=['identity', 'sigmoid', 'tanh', 'ReLU'])
    
    return parser.parse_args()

def evaluate_model(model, X_val, y_val, loss_fn, batch_size):
    total_loss = 0
    correct = 0
    num_batches = max(1, np.ceil(len(X_val) / batch_size))

    
    for i in range(0, len(X_val), batch_size):
        batch_X = X_val[i:i+batch_size]
        batch_y = y_val[i:i+batch_size]
        
        outputs = model.forward(batch_X)
        loss = loss_fn(outputs, batch_y)
        
        total_loss += loss
        correct += accuracy(outputs, batch_y)
    
    avg_loss = total_loss / num_batches
    avg_acc = correct / num_batches
    return avg_loss, avg_acc

def train_model(args):  
    config = args
    run_name = "Train"
    
    wandb.init(project=config.wandb_project, name=run_name)
    
    X_train, y_train, X_test, y_test = load_data(config.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)
    # print("Sample y_train:", y_train[:5])
    # print("Sample y_val:", y_val[:5])

    
    model = FeedForwardNN(
        input_size=784,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        activation=config.activation,
        weight_init=config.weight_init
    )
    
    loss_fn, _ = get_loss(config.loss)
    optimizer = get_optimizer(config.optimizer, model.parameters(), config)
    num_batches = max(1, np.ceil(len(X_train) / config.batch_size))
    
    for epoch in range(config.epochs):
        total_loss = 0 
        correct = 0
        for i in range(0, len(X_train), config.batch_size):
            batch_X = X_train[i:i+config.batch_size]
            batch_y = y_train[i:i+config.batch_size]
            
            outputs = model.forward(batch_X)
            loss = loss_fn(outputs, batch_y)
            
            model.backward(batch_y)
            optimizer.step()
            
            total_loss += loss
            correct += accuracy(outputs, batch_y)
        
        avg_loss = total_loss / num_batches
        avg_acc = correct / num_batches
        
        val_loss, val_acc = evaluate_model(model, X_val, y_val, loss_fn, config.batch_size)
        
        wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': avg_acc, 'val_loss': val_loss, 'val_accuracy': val_acc})
        
        print(f'Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if epoch == config.epochs - 1:
            wandb.run.summary["Train Loss"] = avg_loss
            wandb.run.summary["Train Accuracy"] = avg_acc
            wandb.run.summary["Val Loss"] = val_loss
            wandb.run.summary["Val Accuracy"] = val_acc
            
            # Save best model weights to a file by runing the train.py again with best parameters as the default args
            #Comment out only when train.py is running not sweep.py
            model_weights = {
                "weights": model.weights,
                "biases": model.biases
            }
            np.save("model_weights.npy", model_weights)
    
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train_model(args)
