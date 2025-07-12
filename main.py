from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# filter warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/penguins.csv')

print(df.sample(5, random_state=42))

df.info()

# З колонки Non-Null Count бачимо, що тільки декілька рядків даних мають пропущені значення. Можемо видалити їх з датасету.

df = df.dropna().reset_index(drop=True)
df.info()

# перевіряємо розподіл видів в БД

plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='species')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Target feature distribution")

plt.tight_layout()
plt.show()

# Подивимось розподіл категоріальної змінної island.

plt.figure(figsize=(4,3))
ax = sns.countplot(data=df, x='island')
for i in ax.containers:
    ax.bar_label(i)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
            
plt.suptitle("Island feature distribution")

plt.tight_layout()
plt.show()

# Подивимось на попарний розподіл числових ознак.

plt.figure(figsize=(6,6))
sns.pairplot(data=df, hue='species').fig.suptitle('Numeric features distribution', y=1)
plt.show()

# Підготовка ознак моделі
features = ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

df = df.loc[:, features]

df.loc[df['species']=='Adelie', 'species']=0
df.loc[df['species']=='Gentoo', 'species']=1
df.loc[df['species']=='Chinstrap', 'species']=2
df = df.apply(pd.to_numeric)

df.head(2)

# Представимо матрицю ознак X та вектор таргетової змінної y як numpy-масив.

X = df.drop('species', axis =1).values
y = df['species'].values
print("df.drop('species', axis =1).values", X)

# Бачимо, що ознаки набору даних дуже розрізнені в своєму числовому представленні. Щоб гарантувати, що ознаки будуть представлені в одному масштабі, використаємо StandardScaler.

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("scaler.fit_transform(X)", X)

# Розділимо дані на тренувальні та тестові.
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42, test_size =0.33, stratify=y)

# Для подальшої роботи з інструментами фреймворку PyTorch перетворимо дані з numpy-масивів у torch.tensor.

X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).long()

X_test = torch.Tensor(X_test).float()
y_test = torch.Tensor(y_test).long()

print(X_train[:1])
print(y_train[:10])

# Для матриці ознак X ми зберегли значення з плаваючою комою, у т.ч. для вектору таргету y зберегли цілочисельний вигляд. Використання self.long() є еквівалентом self.to(torch.int64), тобто приведення до цілочисельного типу даних.

#  💡 Модифіковані функції Leaky ReLU, Parametric ReLU (PReLU) і Exponential Linear Unit (ELU) також мають властивості, які сприяють запобіганню згасання градієнта. Наприклад, Leaky ReLU додає невеликий нахил для негативних значень, а ELU має експоненційну залежність негативних значень.

# Створимо клас нейронної мережі для вирішення задачі багатокласової класифікації.

class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=20, out_dim=3):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            
            nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )    
        
    def forward(self, x):
        output = self.features(x)
        return output
    
    # Отже, маємо нейронну мережу, яка складається з лінійного шару, шару активації ReLU, фінального лінійного шару та шару Softmax.

    # Softmax — це функція активації, яка використовується в машинному навчанні, найчастіше в задачах класифікації, для перетворення вектору з невіднормованими виходами нейронної мережі (логітами, logits) у вектор ймовірностей. Вона забезпечує, щоб сума ймовірностей для всіх класів дорівнювала 1.

    # Ініціалізуємо модель.
model = LinearModel(X_train.shape[1], 20, 3)

'''
Кількість вхідних нейронів in_dim буде дорівнювати кількості ознак матриці 
X. Проміжний шар матиме 20 нейронів. Вихідний шар матиме кількість нейронів, рівних кількості класів - 3.
Виходячи з EDA, для вирішення цієї задачі буде достатньо невеликої нейронної мережі з невеликою кількістю прихованих шарів.

В якості функції втрат використаємо Cross-Entropy Loss.
💡 Cross-entropy є однією з найпопулярніших функцій втрат для задач багатокласової класифікації. Вона оцінює ефективність моделі, порівнюючи передбачувані ймовірності класів з фактичними мітками.

Недоліки функції втрат Cross-Entropy Loss

- Чутливість до неправильних передбачень. Модель може бути сильно “оштрафована” за неправильні передбачення, особливо якщо вона впевнено передбачає неправильний клас.
- Потреба у нормалізації. Щоб коректно застосувати функцію крос-ентропії, передбачення повинні бути нормалізовані до ймовірностей (наприклад, через Softmax).
'''
criterion = nn.CrossEntropyLoss()
# В якості оптимізаційного алгоритму використаємо Stochastic Gradient Descent (SGD).
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Кількість епох визначимо рівною 400.
num_epoch = 400 
# Створимо декілька об’єктів-списків, у які будемо зберігати результати тренування: accuracy та loss.
train_loss = []
test_loss = []

train_accs = []
test_accs = []

# Тренування
model.train()
# Це дає можливість обчислювати градієнти моделі, оновлювати ваги, а також активує ряд шарів, що не є активними в режимі валідації, наприклад Dropout та BatchNormalization (про них пізніше).
# - Виконуємо forward pass, отримуємо результати передбачення моделі.

outputs = model(X_train)

# - Розраховуємо значення функції втрат та зберігаємо результат для подальшої обробки.
loss = criterion(outputs, y_train)    
train_loss.append(loss.cpu().detach().numpy())

# - Робимо крок оптимізації.

optimizer.zero_grad()    
loss.backward()
optimizer.step()

# - Розраховуємо точність моделі та зберігаємо результат.

acc = 100 * torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
train_accs.append(acc)

# Тестування
# - Переведемо модель в режим валідації.

model.eval()

# 💡 В режимі валідації ми не обраховуємо градієнти та не здійснюємо зворотне розповсюдження помилки. Також деякі шари, як наприклад Dropout та BatchNormalization, втрачають свої властивості. Це дає можливість мінімізувати розрахункові ресурси, витрати пам’яті, пришвидшує виконання обрахунків і не дає моделі доступ до тестових даних.

# - Робимо forward pass та розраховуємо функцію втрат, але не робимо крок оптимізації. Розраховуємо точність на тестових даних.

outputs = model(X_test)
        
loss = criterion(outputs, y_test)
test_loss.append(loss.cpu().detach().numpy())
        
acc = 100 * torch.sum(y_test==torch.max(outputs.data, 1)[1]).double() / len(y_test)
test_accs.append(acc)

for epoch in range(num_epoch):
    
    # train the model
    model.train()
    
    outputs = model(X_train)
    
    loss = criterion(outputs, y_train)    
    train_loss.append(loss.cpu().detach().numpy())
    
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
    
    acc = 100 * torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
    train_accs.append(acc)
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                       %(epoch+1, num_epoch, loss.item(), acc.item()))
        
    # test the model
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        
        loss = criterion(outputs, y_test)
        test_loss.append(loss.cpu().detach().numpy())
        
        acc = 100 * torch.sum(y_test==torch.max(outputs.data, 1)[1]).double() / len(y_test)
        test_accs.append(acc)

# Аналіз результатів
plt.figure(figsize=(4, 3))
plt.plot(train_loss, label='Train')
plt.plot(test_loss, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Validation')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Metric')
plt.show()

