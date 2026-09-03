import pandas as pd
import numpy as np
import ydf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

train_df = pd.read_csv("C:\\Users\\Hmbb7\\Desktop\\kaggle\\train.csv")
test_df=pd.read_csv('C:\\Users\\Hmbb7\\Desktop\\kaggle\\test.csv')

print("train_shape:",train_df.shape)
print("test_shape:",test_df.shape)

print(train_df.head(3))

print(train_df['SalePrice'].describe())

train_df.info()

train_df=train_df.drop('Id',axis=1)
test_ids=test_df.pop('Id')

rng=np.random.default_rng(42)
mask=rng.random(len(train_df))<0.7
val_df=train_df[~mask]
train_split=train_df[mask]
train_split['SalePrice']=np.log1p(train_split['SalePrice'])
print('练习册:',train_split.shape,'模拟考:',val_df.shape)

learner=ydf.RandomForestLearner(
    label='SalePrice',
    task=ydf.Task.REGRESSION,
    num_trees=300,
)

model=learner.train(train_split)

evaluation=model.evaluate(val_df)

print("验证集评估结果：\n",evaluation)
print("OOB RMSE",model.self_evaluation().rmse)


importance=model.variable_importances()['NUM_AS_ROOT']

for score,name in importance[:8]:
    print(f"{name}被选作根节点{int(score)}次")

names=[n for _, n in importance[:10]][::-1]
scores=[s for s,_ in importance[:10]][::-1]

plt.barh(names,scores)
plt.title("哪些属性最能决定房价（被选作根节点次数）")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.show()


preds=model.predict(test_df)
preds1=np.expm1(preds)
submission=pd.DataFrame({'Id':test_ids,'SalePrice':preds1.round(0)})
submission.to_csv(r"C:\Users\Hmbb7\Desktop\kaggle\submission1.csv",index=False)
print("预测结果已保存到submission.csv")

