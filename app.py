# global imports 
import streamlit as st 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

# function definitions 
def capitalize_columns(data):
    '''
    Capitalize the column names of the input data
    '''
    data.columns = data.columns.str.capitalize()
    return data

def manipulate_data(data):
    '''
    Assign a numerical value to the feature `sex`
    Use one-hot encoding on the feature `pclass`
    Fill in the missing values in the `age` column
    Only select the required features 
    '''
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})
    pclass = pd.get_dummies(data['Pclass']).rename(columns= {1:'FirstClass', 2:'SecondClass', 3:'ThirdClass'})
    data = pd.concat([data, pclass], axis=1)
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    return data[['Age','Sex','FirstClass','SecondClass','ThirdClass','Survived']]

# import data 
train_df = pd.read_csv('data/train.csv')  

# clean data 
train_df = capitalize_columns(train_df)  
train_df = manipulate_data(train_df)

# split data 
features=['Age','Sex','FirstClass','SecondClass','ThirdClass']
X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df['Survived'], test_size=0.2, random_state=42)

# scale data
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# build the model
model = LogisticRegression()
model.fit(train_features, y_train)
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)

# confusion matrix
y_pred = model.predict(test_features)
cm = confusion_matrix(y_test, y_pred)
print(cm)

TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

# display confusion matrix using streamlit
st.title("Titanic Survival Prediction")
st.subheader('Train score: {:.3f}'.format(train_score))
st.subheader('Test score: {:.3f}'.format(test_score))
st.table(train_df.head(5))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['True Positive', 'False Negative', 'False Positive', 'True Negative'], [TP, FN, FP, TN])    
ax.set_xlabel('Confusion Matrix')
ax.set_ylabel('Count')
st.pyplot(fig)

# user input
name = st.text_input("Name of the Passanger")
sex = st.selectbox("Sex", options=['Male', 'Female'])
sex = 0 if sex == "Male" else 1
age = st.slider('Age', 1, 100, 1)
p_class = st.selectbox("Pclass", options=['First Class', 'Second Class', 'Third Class'])
f_class, s_class, t_class = 0, 0, 0
if p_class == 'First Class':
    f_class = 1
elif p_class == 'Second Class':
    s_class = 1
else:
    t_class = 1

# scale user input 
input_data = scaler.transform([[age, sex, f_class, s_class, t_class]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)

if prediction == 0:
    st.subheader('Passenger {} is likely to not have survived with a probability of {}'.format(name, round(predict_probability[0][0]*100, 3)))
else:
    st.subheader('Passenger {} is likely to have survived with a probability of {}'.format(name, round(predict_probability[0][1]*100, 3)))
