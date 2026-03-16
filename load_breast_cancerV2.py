import torch
from sklearn.datasets import load_breast_cancer
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)

y_train_tensor = y_train_tensor.unsqueeze(1)
y_test_tensor = y_test_tensor.unsqueeze(1)

criterion = nn.BCEWithLogitsLoss()
optimaizer = optim.Adam(model.parameters(),lr = 0.01)
for epoch in range(510):

    optimaizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output,y_train_tensor)
    loss.backward()
    optimaizer.step()

    if epoch %10 == 0:
        print(f'epochs:{epoch}',f'loss: {loss.item():.3f}')

with torch.no_grad():
    output = model(X_test_tensor)
    predictions = (torch.sigmoid(output) > 0.5).float()
    accuracy = (predictions == y_test_tensor).float().mean()
    print(f'accuracy:{accuracy.item()*100:.3f}%')