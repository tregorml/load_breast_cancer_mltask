from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc

    print(f"{model_name}:")
    print(f"  Train: {train_acc:.3f}")
    print(f"  Test:  {test_acc:.3f}")
    print(f"  Gap:   {gap:.3f}")
    print()

    return {
        'name': model_name,
        'train': train_acc,
        'test': test_acc,
        'gap': gap
    }

titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = titanic.frame

columns_to_keep = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']
df = df[columns_to_keep].copy()

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['sex'] = df['sex'].astype(int)
df['survived'] = df['survived'].astype(int)
df = df.dropna()

X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = [
    (xgb.XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=10, random_state=42),
     "XGBoost"),

    (LogisticRegression(max_iter=1000),
     "Logistic Regression"),

    (DecisionTreeClassifier(max_depth=5),
     "Decision Tree"),

    (RandomForestClassifier(max_depth=4, n_estimators=100, random_state=42),
     "Random Forest"),

    (lgb.LGBMClassifier(max_depth=4, n_estimators=100,random_state=42 ,verbose=-1),
     'Lightgbm'),

    (CatBoostClassifier(max_depth=4, n_estimators=100,random_state=42 ,verbose=False),
     'CatBoost')
]

results = []
for model, name in models:
    result = evaluate_model(model, name, X_train, X_test, y_train, y_test)
    results.append(result)

print("===================================")
print("SUMMARY")
print("===================================")

best_test = max(results, key=lambda x: x['test'])
worst_test = min(results, key=lambda x: x['test'])
lowest_gap = min(results, key=lambda x: x['gap'])

print(f"Best test accuracy:  {best_test['name']} ({best_test['test']:.3f})")
print(f"Worst test accuracy: {worst_test['name']} ({worst_test['test']:.3f})")
print(f"Lowest gap:          {lowest_gap['name']} ({lowest_gap['gap']:.3f})")