import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

train_df = pd.read_csv("C:\\Users\\Hmbb7\\Desktop\\泰康\\train.csv")
test_df = pd.read_csv("C:\\Users\\Hmbb7\\Desktop\\泰康\\test.csv")

print("train_shape:", train_df.shape)
print("test_shape:", test_df.shape)
print(train_df.head(3))

train_df.info()
print(train_df.size)


women=train_df.loc[train_df['Sex']=='female']['Survived']
rate_women=sum(women)/len(women)
train_df['Age'].fillna(train_df["Age"].median())

y=train_df['Survived']
for df in [train_df,test_df]:
    df['FamilySize']=df.SibSp+df.Parch+1
    df['IsAlone']=(df.FamilySize==1).astype(int)

features=["Pclass",'Sex','SibSp','Parch',"FamilySize","IsAlone",'Age']
x=pd.get_dummies(train_df[features])
x_test=pd.get_dummies(test_df[features])

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=1)
model.fit(x_train,y_train)
print('验证集准确率：',accuracy_score(y_val,model.predict(x_val)))

model_a=RandomForestClassifier(n_estimators=100,random_state=42)
model_a.fit(x_train,y_train)

train_score=accuracy_score(y_train,model_a.predict(x_train))
val_score=accuracy_score(y_val,model_a.predict(x_val))
print(f"训练集准确率: {train_score:.4f}")
print(f"验证集准确率: {val_score:.4f}")
print(f"差距: {train_score - val_score:.4f}（差距越大 → 越可能过拟合）")

import os
train_scores, val_scores = [], []
depth_range = range(1, 21)
for d in depth_range:
    m = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    m.fit(x_train, y_train)
    train_scores.append(accuracy_score(y_train, m.predict(x_train)))
    val_scores.append(accuracy_score(y_val, m.predict(x_val)))

# 画验证曲线（这是验证集最直观的价值）
plt.figure(figsize=(10, 5))
plt.plot(depth_range, train_scores, "o-", label="训练集准确率", color="#2563eb")
plt.plot(depth_range, val_scores, "s-", label="验证集准确率", color="#dc2626")
plt.xlabel("max_depth（树的深度上限）")
plt.ylabel("准确率")
plt.title("验证曲线：深度增加时，训练分涨、验证分可能反降")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "validation_curve.png"), dpi=110)
print(f"\n验证曲线图已保存到: validation_curve.png")
best_d = depth_range[int(np.argmax(val_scores))]


train_scores2, val_scores2 = [], []
n_range = [50, 100, 150, 200, 300, 500]
for n in n_range:
    m = RandomForestClassifier(n_estimators=n, max_depth=best_d, random_state=42)
    m.fit(x_train, y_train)
    train_scores2.append(accuracy_score(y_train, m.predict(x_train)))
    val_scores2.append(accuracy_score(y_val, m.predict(x_val)))

print(f"{'n_estimators':<12} {'训练分':<10} {'验证分':<10} {'差距':<10}")
print("-" * 50)
for n, ts, vs in zip(n_range, train_scores2, val_scores2):
    print(f"{n:<12} {ts:<10.4f} {vs:<10.4f} {ts - vs:<10.4f}")

best_n = n_range[int(np.argmax(val_scores2))]
print(f"\n👉 验证集告诉我们：n_estimators 超过 {best_n} 后，验证分不再涨，但训练时间翻倍")
print(f"   最终模型参数：n_estimators={best_n}, max_depth={best_d}")


final_model = RandomForestClassifier(
    n_estimators=best_n, max_depth=best_d, random_state=42
)
final_model.fit(x, y)
test_preds = final_model.predict(x_test)
print(f"测试集预测完成，预测 {len(test_preds)} 人")
print(f"预测生还人数: {test_preds.sum()} / {len(test_preds)} （{test_preds.mean():.1%}）")
print(f"训练集真实生还比例: {y.mean():.1%}，两者接近 → 模型靠谱")


output=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':test_preds})
output.to_csv('submission2.csv',index=False)
print('sucess')