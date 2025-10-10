import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -----------------------------
# Define HeartNet Model
# -----------------------------
class HeartNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# -----------------------------
# Generate Synthetic Dataset
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)
N = 5000   # Larger dataset for better training

X = np.zeros((N, 12))
X[:,0] = np.random.randint(40, 90, N)     # Age
X[:,1] = np.random.randint(0,2,N)         # Anaemia
X[:,2] = np.random.randint(50,1000,N)     # Creatinine phosphokinase
X[:,3] = np.random.randint(0,2,N)         # Diabetes
X[:,4] = np.random.randint(15,60,N)       # Ejection fraction
X[:,5] = np.random.randint(0,2,N)         # High blood pressure
X[:,6] = np.random.randint(100000,400000,N) # Platelets
X[:,7] = np.random.uniform(0.5,3.0,N)     # Serum creatinine
X[:,8] = np.random.randint(120,150,N)     # Serum sodium
X[:,9] = np.random.randint(0,2,N)         # Sex
X[:,10] = np.random.randint(0,2,N)        # Smoking
X[:,11] = np.random.randint(5,100,N)      # Time (days in hospital)

# -----------------------------
# Generate Balanced Labels
# -----------------------------
risk_score = (
    (X[:,4]<35).astype(int) +    # low EF
    (X[:,7]>1.5).astype(int) +   # high serum creatinine
    (X[:,8]<135).astype(int) +   # low sodium
    (X[:,0]>70).astype(int) +    # elderly
    (X[:,5]==1).astype(int) +    # hypertension
    (X[:,3]==1).astype(int)      # diabetes
)

# Label: >=3 risk factors = 1 (high risk), else 0 (low risk)
y = (risk_score >= 3).astype(int)

# Balance dataset by undersampling the majority class
pos_idx = np.where(y==1)[0]
neg_idx = np.where(y==0)[0]
min_count = min(len(pos_idx), len(neg_idx))
sel_pos = np.random.choice(pos_idx, min_count, replace=False)
sel_neg = np.random.choice(neg_idx, min_count, replace=False)
sel_idx = np.concatenate([sel_pos, sel_neg])
X = X[sel_idx]
y = y[sel_idx]

print(f"Dataset size: {len(X)} samples (balanced {sum(y==0)} vs {sum(y==1)})")

# -----------------------------
# Preprocess (Scaling)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Xt = torch.tensor(X_scaled, dtype=torch.float32)
yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

# -----------------------------
# Train Model
# -----------------------------
model = HeartNet(input_dim=12)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 250   # Longer training for stability
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(Xt)
    loss = criterion(out, yt)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 25 == 0:
        pred = (out.detach().numpy() > 0.5).astype(int)
        acc = (pred == y.reshape(-1,1)).mean()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.3f}")

# -----------------------------
# Save Model + Scaler
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/model_initial.pt")
joblib.dump(scaler, "artifacts/scaler.pkl")

print("✅ Training complete. Model and scaler saved in artifacts/ folder.")
