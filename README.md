## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
  import pandas as pd
  import numpy as np
  from scipy import stats
  df=pd.read_csv("data.csv")
  df
  ```
  ![image](https://github.com/user-attachments/assets/2313cd4b-db41-41b6-8c1d-2f5469132b10)
  ```
  from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
  climate=['Cold','Warm','Hot','Very Hot']
  ele=OrdinalEncoder(categories=[climate])
  ele.fit_transform(df[["Ord_1"]])
  ```
  ![image](https://github.com/user-attachments/assets/1b3c9f34-d2c9-4882-bafe-2de46ef8c21d)
  ```
  df['bo2']=ele.fit_transform(df[['Ord_1']])
  df
  ```
  ![image](https://github.com/user-attachments/assets/d8503a8b-7062-4ab3-8d53-96e92e48adfa)
  ```
le=LabelEncoder()
df2=df.copy()
df2['Ord_2']=le.fit_transform(df2['Ord_2'])
df2
```
![image](https://github.com/user-attachments/assets/50fc3cc6-2af8-4889-abf7-f3103d3ecccc)
```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder()
df3=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df[['City']]))
df2=pd.concat([enc,df3],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/4db62384-9f33-4063-9393-917ef0572396)
```
pd.get_dummies(df,columns=['City'])
```
![image](https://github.com/user-attachments/assets/6be74ff6-0c58-46fe-91e3-b4af521eacd8)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/9f6bcb63-2881-49e4-96f3-b59340664ddb)
```
from category_encoders import BinaryEncoder
```
from category_encoders import BinaryEncoder
```
import pandas as pd
```
df=pd.read_csv("C:\\Users\\priya\\Downloads\\data.csv")
```
be=BinaryEncoder()
```
nd=be.fit_transform(df['Ord_2'])
```
df1=pd.concat([df,nd],axis=1)
```
df1=df.copy()
```
df1
```
![image](https://github.com/user-attachments/assets/90dc7d32-03b0-42d4-aefe-b0420d67d39a)
```
from category_encoders import TargetEncoder
```
te=TargetEncoder()
```
cc=df.copy()
```
new=te.fit_transform(X=cc["City"],y=cc["Target"])
```
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/dffb9424-60f1-46e9-897a-e40e3c182823)
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/6ccca6df-60cc-48ce-9a9a-1e33d27bf7ee)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/6b56ede2-c09b-4d1f-9979-4edfc2ea49b2)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/aa53ccfe-4fce-4a99-8dfb-5177b2801421)
```
np.reciprocal(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ee086a44-5d08-4b04-8463-9fb5d5e4cce3)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/520ae0d3-8a6c-4482-8702-f0433ef735ce)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f5bd85d9-ebfe-467e-aae0-c50402fe627d)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/7fb6ae6c-1d06-41a9-97e2-78081636cfb7)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
```
from sklearn.preprocessing import QuantileTransformer
```
qt=QuantileTransformer(output_distribution='normal')
```
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/99d30b9d-8883-495c-8adc-92c397db3645)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a3677edf-eb67-43ba-b9ec-de4aae746fd7)
```
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
```
![image](https://github.com/user-attachments/assets/2e9cb7a7-b7b4-4096-b57b-fa5e37a8d51a)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/409c7073-844f-4d5b-b71d-a90b4c326ad7)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b5631112-ce82-4e62-a86a-e291d98dd07f)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/user-attachments/assets/5c954d69-55fe-40be-94ce-4715666e6f03)
# RESULT:
Thus the given data,Feature Encoding,Transformation process and save the data to a file was performed successfully.

       
