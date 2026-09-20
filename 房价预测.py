#导入工具
import pandas as pd
import numpy as np
import ydf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
#读数据
train_df = pd.read_csv("C:\\Users\\Hmbb7\\Desktop\\kaggle\\train.csv")
test_df=pd.read_csv('C:\\Users\\Hmbb7\\Desktop\\kaggle\\test.csv')

print("train_shape:",train_df.shape)
print("test_shape:",test_df.shape)
#数据体检（EDA）
print(train_df.head(3))

print(train_df['SalePrice'].describe())

train_df.info()
#删掉没用的ID列
train_df=train_df.drop('Id',axis=1)
test_ids=test_df.pop('Id')
#划分练习与模拟集
rng=np.random.default_rng(42)  # 随机数发生器，42=种子，保证每次运行结果一样
mask=rng.random(len(train_df))<0.7 #len()多少行，生成多少行的随机数，rng.random(1000)：生成 1000 个 0~1 之间的均匀随机浮点数
val_df=train_df[~mask]
train_split=train_df[mask]
train_split['SalePrice']=np.log1p(train_split['SalePrice']) #取对数
print('练习册:',train_split.shape,'模拟考:',val_df.shape)
#建模型+训练
learner=ydf.RandomForestLearner(
    label='SalePrice',
    task=ydf.Task.REGRESSION, #任务为回归，漏写为分类
    num_trees=300,
)

model=learner.train(train_split)
#评估
evaluation=model.evaluate(val_df)

print("验证集评估结果：\n",evaluation)
##均方根误差
print("OOB RMSE",model.self_evaluation().rmse)

#特征重要性
# NUM_AS_ROOT = 有多少棵树把这个特征选作第一个问题（根节点）
importance=model.variable_importances()['NUM_AS_ROOT']

for score,name in importance[:8]:
    print(f"{name}被选作根节点{int(score)}次")
#绘图
names=[n for _, n in importance[:10]][::-1] ## [::-1] 倒序，让第一名画在最上面
scores=[s for s,_ in importance[:10]][::-1]

plt.barh(names,scores)
plt.title("哪些属性最能决定房价（被选作根节点次数）")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.show()

#预测
preds=model.predict(test_df)
preds1=np.expm1(preds) #对数还原
submission=pd.DataFrame({'Id':test_ids,'SalePrice':preds1.round(0)})
submission.to_csv(r"C:\Users\Hmbb7\Desktop\kaggle\submission1.csv",index=False)
print("预测结果已保存到submission.csv")

