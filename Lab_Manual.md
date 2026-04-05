## Program 1 

```python
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import calendar

df = pd.read_csv('AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month']) 

df.set_index('Month', inplace=True)

df['12_ma'] = df['Passengers'].rolling(window=12).mean()
print(df.head(15))

plt.figure(figsize=(10,5))
plt.plot(df['Passengers'], label='Actual')
plt.plot(df['12_ma'], color='red', label='12 month avg')
plt.title('Air Passenger Trend Analysis')
plt.legend()
plt.tight_layout()
plt.show()

df['Month_name'] = df.index.month_name()

months = list(calendar.month_name)[1:]
plt.figure(figsize=(10,5))
sns.boxplot(data=df, order=months, x='Month_name', y='Passengers',hue='Month_name', palette='viridis' ,legend=False)
plt.title('Seasonal Box Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Program 2 

```python
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats

iris = pd.read_csv('iris.csv')
iris.head()

iris['sepal_r'] = iris['sepal_length']/iris['sepal_width']
iris['petal_r'] = iris['petal_length']/iris['petal_width']

corr = iris.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Corr M')
plt.show()

g = sns.PairGrid(iris, height=2.5)
g.map_diag(sns.histplot, kde=True)
g.map_upper(sns.scatterplot)
g.map_lower(sns.regplot)
g.add_legend()
plt.show()

plt.figure(figsize=(7,5))
sns.regplot(x='sepal_length', y='petal_length', data=iris, order=3)
plt.title('Reg Plot')
plt.show()

cor_val, p_val = stats.pearsonr(iris['sepal_length'], iris['petal_length'])
round(cor_val,6)
round(p_val,6)
```

## Program 3 

```python
import plotly.express as px 
import pandas as pd 

df = pd.read_csv('gapminder.csv')
df.head()

px.line(df.query("continent == 'Oceania'"), x='year', y='lifeExp', color='country',  title='Oceania Life exp').show()

px.bar(df.query("year == 2007 and continent == 'Europe'"), x='country', y='gdpPercap', color='gdpPercap' ,title='Europe 2007 GDP').show()

px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color='continent', log_x=True, size_max=60, title='health vs wealth').show()
```

## Program 4 

```python
import plotly.express as px 
import pandas as pd 
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv('tips.csv')
app = Dash(__name__)
df.head()

app.layout = html.Div([
    dcc.Dropdown(id ='day', options=df['day'].unique(), value='Sun'), 
    dcc.Dropdown(id='time', options=df['time'].unique(), value='Dinner'), 
    dcc.Graph(id='g1'),dcc.Graph(id='g2'), dcc.Graph(id='g3'), dcc.Graph(id='g4')
])

@app.callback(
    [Output('g1', 'figure'), Output('g2', 'figure'), Output('g3', 'figure'), Output('g4', 'figure')], 
    [Input('day', 'value'), Input('time', 'value')]
)

def update(d, t): 
    f = df[(df.day == d) & (df.time == t)]
    return (
        px.scatter(f, x='total_bill', y='tip', color='sex'), 
        px.bar(f, x='sex', y='tip', color='smoker'), 
        px.histogram(f, x='total_bill', color='sex'), 
        px.box(f, x='sex', y='tip', color='sex') 
    )

if __name__ == '__main__': 
    app.run(debug=True, port=5000)
```

## Program 5 

```python
import pandas as pd 
import plotly.express as px 

df = pd.read_csv('earthquakes-23k.csv')
df.head()

df.dropna(subset=['Latitude', 'Longitude', 'Magnitude'])

px.scatter_geo(df, lat='Latitude', lon='Longitude', size='Magnitude', color='Magnitude', projection='natural earth', hover_name='Magnitude').show()
``` 

## Program 6 

```python
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='.2f')
plt.title('Conf Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[1,0],'--')
plt.title('ROC Curve')
plt.legend()
plt.show()

imp = model.feature_importances_
feats = pd.Series(imp, index=x.columns).sort_values(ascending=False)

feats.head(10).plot(kind='barh')
plt.title('Feat Importance')
plt.show()
```


