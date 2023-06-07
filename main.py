import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#1
import pandas_datareader as web
import matplotlib.pyplot as plt
start_date = '2017-01-01'
end_date = '2018-12-31'
df = web.DataReader('AAPL', 'yahoo', start_date, end_date)
plt.plot(df['Close'])
plt.title('Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
#2
df_high = df['High'].to_frame()
Q1 = df_high['High'].quantile(0.25)
Q3 = df_high['High'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_high[(df_high['High'] < Q1 - 1.5 * IQR) | (df_high['High'] > Q3 + 1.5 * IQR)].index
plt.figure(figsize=(10, 6))
plt.plot(df_high.index, df_high['High'], color='blue', label='Цены акций')
plt.plot(df_high.loc[outliers].index, df_high['High'].loc[outliers], 'ro', label='Выбросы')
plt.title('Цены акций Apple')
plt.xlabel('Дата')
plt.ylabel('Максимальная цена, $')
plt.legend()
plt.show()
#Лабораторная работа
#1
data=np.load('average_ratings.npy')
plt.plot(data[0],'r',label='waffle iron french toast')
plt.plot(data[1],'g',label='zwetschgenkuchen bavarian plum cake')
plt.plot(data[2],'purple',label='lime tea')
plt.title('Изменение среднего рейтинга 3 рецептов')
plt.xlabel('Номер дня')
plt.ylabel('Средний рейтинг')
plt.legend(loc='upper left')
plt.show()
ratings = np.load('average_ratings.npy')
import datetime

start_date = datetime.date(2019, 1, 1)
end_date = start_date + datetime.timedelta(days=len(ratings)-1)
dates = pd.date_range(start_date, end_date)

plt.figure(figsize=(10, 6))
plt.plot(dates, ratings[:,0], label='Рецепт 1', linewidth=2)
plt.plot(dates, ratings[:,1], label='Рецепт 2', linewidth=2)
plt.plot(dates, ratings[:,2], label='Рецепт 3', linewidth=2)
plt.xlabel('Дата')
plt.ylabel('Средний рейтинг')
plt.title('Временные ряды среднего рейтинга')
plt.legend()
plt.show()
#2
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
dates=pd.date_range(start='01.01.2019',end='30.12.2021 ',freq='D')
time_series = pd.to_datetime(pd.Series(dates))
data=np.load('average_ratings.npy')
plt.plot(dates)
ax.plot(data[0],'r',label='waffle iron french toast')
ax.plot(data[1],'g',label='zwetschgenkuchen bavarian plum cake')
ax.plot(data[2],'purple',label='lime tea')
ax.set_xlim(dates)
ax.xticks(time_series)
ax.title('Изменение среднего рейтинга 3 рецептов')
ax.xlabel('Номер дня')
ax.ylabel('Средний рейтинг')
ax.legend(loc='upper left')
ax.xticks(dates)
plt.show()
#4
visitors = np.load('visitors.npy')
plt.subplot(1, 2, 1)
plt.plot(visitors)
plt.title('y(x) = λe^(-λx)')
plt.xlabel('Количество дней с момента акции')
plt.ylabel('Число посетителей')
plt.axhline(y=100, color='red')
plt.text(5, 105, 'y(x) = 100')
plt.subplot(1, 2, 2)
plt.plot(visitors)
plt.yscale('log')
plt.title('y(x) = λe^(-λx)')
plt.xlabel('Количество дней с момента акции')
plt.ylabel('Число посетителей')
plt.axhline(y=100, color='red')
plt.text(5, 100, 'y(x) = 100')
plt.show()
#5
df = pd.read_csv('recipes_sample.csv')
def recipe_group(row):
    if row['minutes'] < 5:
        return 'Короткие'
    elif row['minutes'] < 50:
        return 'Средние'
    else:
        return 'Длинные'

df['group'] = df.apply(recipe_group, axis=1)
grouped = df.groupby('group').agg({'n_steps': 'mean', 'id': 'count'})
grouped.plot(kind='bar', y='n_steps', legend=False)
plt.ylabel('Средняя длительность')
plt.xlabel('Группа рецептов')
plt.subplots()
grouped.plot(kind='pie', y='id', legend=False)
plt.title('Размеры групп рецептов')
plt.axis('equal')
plt.show()
#6
df=pd.read_csv('recipes_sample.csv')
df_filtered = df[(df['submitted'].year == 2008) | (df['review_date'].dt.year == 2009)].copy()
grouped = df_filtered.groupby(df_filtered['review_date'].dt.year)['rating'].value_counts().unstack().fillna(0)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

grouped[2008].plot.hist(ax=axes[0], bins=20)
axes[0].set_xlabel('Рейтинг')
axes[0].set_ylabel('Количество отзывов')
axes[0].set_title('2008 год')

grouped[2009].plot.hist(ax=axes[1], bins=20)
axes[1].set_xlabel('Рейтинг')
axes[1].set_ylabel('')
axes[1].set_title('2009 год')

fig.suptitle('Гистограммы рейтинга отзывов в 2008 и 2009 годах', fontsize=14)

plt.tight_layout()
plt.show()
#7
import seaborn as sns
import pandas as pd
df = pd.read_csv('recipes_sample.csv')
def recipe_group(row):
    if row['minutes'] < 5:
        return 'короткий'
    elif row['minutes'] < 50:
        return 'средний'
    else:
        return 'длинный'
df['recipe_group'] = df.apply(recipe_group, axis=1)
sns.scatterplot(x='n_steps', y='n_ingredients', hue='recipe_group', data=df)
plt.title('Диаграмма рассеяния n_steps и n_ingredients')
plt.show()
#8
import pandas as pd

recipes = pd.read_csv('recipes_sample.csv')
reviews = pd.read_csv('reviews_sample.csv')

df = pd.concat([recipes, reviews])

corr = df[['minutes', 'n_steps', 'n_ingredients', 'rating']].corr()

sns.heatmap(corr, annot=True, cmap="YlOrRd")
plt.title('Корреляционная матрица числовых столбцов таблиц recipes и reviews')
plt.show()