"""
Training utilities for blood cell classification models.
This module contains functions for training, evaluation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
import itertools
import os


def train_model(model, train_gen, valid_gen, epochs=30, model_name="model"):
    """
    Train the model with given generators
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        valid_gen: Validation data generator
        epochs: Number of training epochs
        model_name: Name for saving the model
        
    Returns:
        Training history
    """
    print(f"Training {model_name} for {epochs} epochs...")
    
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        steps_per_epoch=None,
        workers=2,
        verbose=1
    )
    
    print(f"Training completed for {model_name}")
    return history


def plot_training_history(history, model_name="Model", save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Training history from model.fit()
        model_name: Name of the model for plot titles
        save_path: Path to save the plots
    """
    # Extract metrics
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(train_acc) + 1)
    
    # Find best epochs
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_acc_epoch = np.argmax(val_acc) + 1
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, train_loss, 'r-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'b-', label='Validation Loss', linewidth=2)
    ax1.scatter(best_val_loss_epoch, val_loss[best_val_loss_epoch-1], 
               s=100, c='orange', label=f'Best Epoch: {best_val_loss_epoch}', zorder=5)
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_acc, 'r-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'b-', label='Validation Accuracy', linewidth=2)
    ax2.scatter(best_val_acc_epoch, val_acc[best_val_acc_epoch-1], 
               s=100, c='orange', label=f'Best Epoch: {best_val_acc_epoch}', zorder=5)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model(model, test_gen, model_name="Model"):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator
        model_name: Name of the model
        
    Returns:
        Evaluation results dictionary
    """
    print(f"Evaluating {model_name}...")
    
    # Get test data
    X_test, y_test = next(test_gen)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    mae = mean_absolute_error(true_classes, predicted_classes)
    
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'mae': mae,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'X_test': X_test,
        'y_test': y_test
    }
    
    print(f"{model_name} Test Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", 
                         normalize=True, save_path=None):
    """
    Plot confusion matrix with proper formatting
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: List of class names
        model_name: Name of the model
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy and misclassification rate
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        thresh = cm_norm.max() / 2.0
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        thresh = cm.max() / 2.0
    
    plt.title(f'{model_name} - Confusion Matrix{"(Normalized)" if normalize else ""}')
    plt.colorbar()
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm_norm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label\nAccuracy={accuracy:.4f}; Misclassification={misclass:.4f}')
    
    if save_path:
        suffix = "_normalized" if normalize else "_raw"
        plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix{suffix}.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_detailed_metrics(y_true, y_pred, class_names):
    """
    Calculate detailed classification metrics
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: List of class names
        
    Returns:
        Dictionary with detailed metrics
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Calculate rates
    TPR = TP / (TP + FN)  # Sensitivity/Recall
    TNR = TN / (TN + FP)  # Specificity
    PPV = TP / (TP + FP)  # Precision
    NPV = TN / (TN + FN)  # Negative Predictive Value
    FPR = FP / (FP + TN)  # False Positive Rate
    FNR = FN / (TP + FN)  # False Negative Rate
    FDR = FP / (TP + FP)  # False Discovery Rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Accuracy per class
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy_per_class': ACC,
        'precision': PPV,
        'recall': TPR,
        'specificity': TNR,
        'negative_predictive_value': NPV,
        'false_positive_rate': FPR,
        'false_negative_rate': FNR,
        'false_discovery_rate': FDR
    }
    
    return metrics


def print_classification_report(y_true, y_pred, class_names, model_name="Model"):
    """
    Print detailed classification report
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: List of class names
        model_name: Name of the model
    """
    print(f"\n{model_name} - Detailed Classification Report:")
    print("=" * 60)
    
    # Get sklearn classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Get detailed metrics
    metrics = calculate_detailed_metrics(y_true, y_pred, class_names)
    
    print("\nDetailed Metrics per Class:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Accuracy: {metrics['accuracy_per_class'][i]:.4f}")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall'][i]:.4f}")
        print(f"  Specificity: {metrics['specificity'][i]:.4f}")
        print(f"  F1-Score: {2 * metrics['precision'][i] * metrics['recall'][i] / (metrics['precision'][i] + metrics['recall'][i]):.4f}")
        print()


def plot_sample_predictions(X_test, y_true, y_pred, predictions_proba, class_names, 
                           model_name="Model", num_samples=25, save_path=None):
    """
    Plot sample predictions with probabilities
    
    Args:
        X_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        predictions_proba: Prediction probabilities
        class_names: List of class names
        model_name: Model name
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 15))
    
    # Randomly select samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        # Get prediction info
        pred_class = y_pred[idx]
        true_class = y_true[idx]
        confidence = predictions_proba[idx][pred_class]
        
        # Set color based on correctness
        color = 'green' if pred_class == true_class else 'red'
        
        plt.xlabel(f'Pred: {class_names[pred_class]} ({confidence:.2f})\nTrue: {class_names[true_class]}', 
                  color=color, fontsize=10)
        
        # Display image
        image = X_test[idx]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        plt.imshow(image)
    
    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_sample_predictions.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def save_model_and_results(model, history, results, model_name, save_dir="results"):
    """
    Save model, history, and results
    
    Args:
        model: Trained Keras model
        history: Training history
        results: Evaluation results
        model_name: Name of the model
        save_dir: Directory to save results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save history as numpy arrays
    history_path = os.path.join(save_dir, f'{model_name}_history.npz')
    np.savez(history_path, **history.history)
    print(f"Training history saved to: {history_path}")
    
    # Save results
    results_path = os.path.join(save_dir, f'{model_name}_results.npz')
    np.savez(results_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
    print(f"Results saved to: {results_path}")


def compare_models(model_results, model_names, class_names, save_path=None):
    """
    Compare multiple models' performance
    
    Args:
        model_results: List of model results dictionaries
        model_names: List of model names
        class_names: List of class names
        save_path: Path to save comparison plots
    """
    # Create comparison metrics
    metrics_comparison = {
        'Model': model_names,
        'Accuracy': [r['test_accuracy'] for r in model_results],
        'Loss': [r['test_loss'] for r in model_results],
        'MAE': [r['mae'] for r in model_results]
    }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    axes[0].bar(model_names, metrics_comparison['Accuracy'], color=['skyblue', 'lightcoral'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(metrics_comparison['Accuracy']):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Loss comparison
    axes[1].bar(model_names, metrics_comparison['Loss'], color=['lightgreen', 'gold'])
    axes[1].set_title('Model Loss Comparison')
    axes[1].set_ylabel('Loss')
    for i, v in enumerate(metrics_comparison['Loss']):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # MAE comparison
    axes[2].bar(model_names, metrics_comparison['MAE'], color=['plum', 'orange'])
    axes[2].set_title('Model MAE Comparison')
    axes[2].set_ylabel('Mean Absolute Error')
    for i, v in enumerate(metrics_comparison['MAE']):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'models_comparison.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print("=" * 50)
    for i, name in enumerate(model_names):
        print(f"{name}:")
        print(f"  Accuracy: {metrics_comparison['Accuracy'][i]:.4f}")
        print(f"  Loss: {metrics_comparison['Loss'][i]:.4f}")
        print(f"  MAE: {metrics_comparison['MAE'][i]:.4f}")
        print()
    
    return metrics_comparison