from sklearn.metrics import roc_auc_score, mean_squared_error

def evaluate_model(model, val_loader):
    try:
        y_true = []
        y_pred = []
        for images, labels in val_loader:
            outputs = model(images)
            y_pred.extend(outputs.detach().numpy())
            y_true.extend(labels.numpy())
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC: {auc}")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
