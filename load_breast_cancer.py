import torch
from sklearn.datasets import load_breast_cancer
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def readable():
    print('='*80)

data = load_breast_cancer( as_frame=True)
df = data.frame

print(data.frame.columns)
readable()
print(data.frame)
readable()

df['target'] = df['target'].astype(int)
X = df.drop(columns=['target'])
y = df['target']

class CancerPredict(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30,16)
        self.layer2 = nn.Linear(16, 1)
    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)

model = CancerPredict()
print(model)

X_tensor = torch.tensor(X.values,dtype = torch.float32)
y_tensor = torch.tensor(y.values,dtype = torch.float32)
output = model(X_tensor)
y_tensor = y_tensor.unsqueeze(1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled,dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss()
optimaizer = optim.Adam(model.parameters(),lr = 0.01)
for epoch in range(510):

    optimaizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output,y_tensor)
    loss.backward()
    optimaizer.step()

    if epoch %10 == 0:
        print(f'epochs:{epoch}',f'loss: {loss.item():.3f}')