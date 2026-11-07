import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score 

class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.w = None
        self.lr = learning_rate
        self.epochs = epochs

    def softmax(self, X: np.ndarray):
        logits = X @ self.w
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


    def loss_fn(self, y_hat: np.ndarray,  y: np.ndarray):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        return - ((y * np.log(y_hat + 10e-10)).sum(axis=1)).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        K = y.shape[1]
        self.w = np.zeros((D, K))
        loss_value = None

        self.loss_history = []

        with tqdm(range(self.epochs)) as pb:
            for epoch in pb:
                y_hat = self.softmax(X)

                loss_value = self.loss_fn(y_hat, y)
                self.loss_history.append(loss_value)

                grad = X.T @ (y_hat - y) / N
                self.w = self.w - self.lr * grad

                pb.set_description(
                    f"Training | Epoch {epoch + 1}/{self.epochs} | Loss: {loss_value:.4f}"
                )

        return self

    def predict(self, X: np.ndarray):
        logits = self.softmax(X)
        y_predicted = np.argmax(logits, axis=1)

        return y_predicted

    def evaluate(self, X, y) -> dict:
        y_hat = self.predict(X)

        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        P = precision_score(y, y_hat, average='macro')
        R = recall_score(y, y_hat, average='macro')
        F1 = f1_score(y, y_hat, average='macro')

        return {
            "Precision": P,
            "Recall": R,
            "F1 score": F1
        }