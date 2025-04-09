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
NAME : VINODINI R
REGISTER NUMBER : 212223040244
```
```
import pandas as pd
import numpy as np
from scipy import stats
```
```
df=pd.read_csv("C:\\Users\\admin\Downloads\\data.csv")
df
```
<img width="576" alt="a1" src="https://github.com/user-attachments/assets/14f9d2fc-e4fa-43ab-9dd7-4456ea9cf500" />


```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
climate=['Cold','Warm','Hot','Very Hot']
ele=OrdinalEncoder(categories=[climate])
ele.fit_transform(df[['Ord_1']])
```



<img width="576" alt="a2" src="https://github.com/user-attachments/assets/e0b1cdda-ca28-472e-bd09-4d0f676be59b" />

```
df['bo2']=ele.fit_transform(df[['Ord_1']])
df
```

<img width="576" alt="a3" src="https://github.com/user-attachments/assets/40b1b90c-d045-4563-a574-954da6c0c757" />

```
le=LabelEncoder()
df2=df.copy()
df2['Ord_2']=le.fit_transform(df2['Ord_2'])
df2
```


<img width="576" alt="a4" src="https://github.com/user-attachments/assets/ee0f6874-ce17-4c18-b968-b177b4a7a40e" />


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df3=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df[['City']]))
df2=pd.concat([enc,df3],axis=1)
df2
```

<img width="690" alt="a5" src="https://github.com/user-attachments/assets/222e9931-614b-4b75-90b6-8c7d634af398" />


```
pd.get_dummies(df,columns=['City'])
```

<img width="794" alt="a6" src="https://github.com/user-attachments/assets/ce3174c3-540b-450d-9a1e-c38c6ce459e7" />

```
!pip install category_encoders
from category_encoders import BinaryEncoder
dfd=pd.read_csv("/content/data.csv")
dfd
```


<img width="576" alt="a7" src="https://github.com/user-attachments/assets/3a7aff40-fa76-4ce5-a1dc-b7077be314d8" />


```
be=BinaryEncoder()
nd=be.fit_transform(dfd['Ord_2'])
df=pd.concat([dfd,nd],axis=1)
df
```


<img width="576" alt="a8" src="https://github.com/user-attachments/assets/ef1dfd0a-2766-4a99-82ab-e89e53dbf42f" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
pd.concat([cc,new],axis=1)
```


<img width="687" alt="a9" src="https://github.com/user-attachments/assets/80dfbdb3-653d-4aed-82a3-91b59f4b47ac" />

```
vf=pd.read_csv("/content/Data_to_Transform.csv")
vf
```

<img width="711" alt="a10" src="https://github.com/user-attachments/assets/3e9ae0a7-4343-40bf-a397-625ed0a82b9e" />

```
vf.skew()
```


<img width="576" alt="a11" src="https://github.com/user-attachments/assets/66b98a01-377a-4a1c-bedf-b3816d967604" />

```
np.log(vf["Highly Positive Skew"])
```


<img width="576" alt="a12" src="https://github.com/user-attachments/assets/7029bd4f-0363-447b-8d61-6549b912bd16" />

```
np.reciprocal(vf["Highly Positive Skew"])
```


<img width="576" alt="a13" src="https://github.com/user-attachments/assets/527c808d-13fc-4002-84b9-85fef4ecb09d" />


```
np.reciprocal(vf["Moderate Positive Skew"])
```

<img width="576" alt="a14" src="https://github.com/user-attachments/assets/c9c2ce92-8ecd-409b-bb1a-445e10d21652" />


```
np.square(vf["Highly Positive Skew"])
```


<img width="576" alt="a15" src="https://github.com/user-attachments/assets/80586186-973c-4409-8d6d-988de121cd23" />

```
vf["Highly Positive Skew"],parameters=stats.boxcox(vf["Highly Positive Skew"])
vf
```
<img width="808" alt="a16" src="https://github.com/user-attachments/assets/b8ce5a76-03bc-48fe-9896-697541f9c0b7" />

```
vf["Moderate Negative Skew_teojohnson"],parameters=stats.yeojohnson(vf["Moderate Negative Skew"])
vf
```


<img width="618" alt="a17" src="https://github.com/user-attachments/assets/53776c9a-691c-4918-ab37-53baa8d2aa5b" />

```
from sklearn.preprocessing import QuantileTransformer
Qt=QuantileTransformer(output_distribution='normal')
vf["Moderate Negative Skew_1"]=Qt.fit_transform(vf[["Moderate Negative Skew"]])
vf
```


<img width="933" alt="a18" src="https://github.com/user-attachments/assets/f685f7b0-23c9-4d8e-bf1a-379e3dc72b1b" />

```
import matplotlib.pyplot as plt
import seaborn as sna 
import statsmodels.api as sm
import scipy.stats as stats
```
```
sm.qqplot(vf["Moderate Negative Skew"],line='45')
plt.show()
```


<img width="576" alt="a19" src="https://github.com/user-attachments/assets/39139240-ce23-4b4a-beb4-8eefb3aeeda9" />

```
sm.qqplot(vf["Moderate Negative Skew_1"],line='45')
plt.show()
```


<img width="576" alt="a20" src="https://github.com/user-attachments/assets/c5a58485-a10f-4833-9f94-0bd9ed494f0d" />

```
vf["Highly Negative Skew_1"]=Qt.fit_transform(vf[["Highly Negative Skew"]])
sm.qqplot(vf["Highly Negative Skew"],line='45')
plt.show()
```


<img width="576" alt="a21" src="https://github.com/user-attachments/assets/39d2f3bf-7c2d-435a-bc9e-c831f16e7f87" />

```
sm.qqplot(vf["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="576" alt="a22" src="https://github.com/user-attachments/assets/52e3fb2d-2df6-4d9a-9b95-ca954333c770" />

```
sm.qqplot(np.reciprocal(vf["Moderate Negative Skew_1"]),line='45')
plt.show()
```


<img width="576" alt="a23" src="https://github.com/user-attachments/assets/303d5376-27bd-4657-8371-3270deac285a" />

```
sm.qqplot(np.abs(vf["Highly Negative Skew_1"]),line='45')
plt.show()
```
<img width="576" alt="a24" src="https://github.com/user-attachments/assets/3523f1a9-46bf-43fd-aeaa-0ab004c6f239" />



```
sm.qqplot(np.log(vf["Highly Negative Skew_1"]),line='45')
plt.show()
```



<img width="628" alt="a25" src="https://github.com/user-attachments/assets/3f72ce95-4772-4b4e-9d76-970cccc2f45e" />

```
sm.qqplot(np.sqrt(vf["Highly Negative Skew_1"]),line='45')
plt.show()
```

<img width="576" alt="a26" src="https://github.com/user-attachments/assets/605aea61-644d-4c72-9be8-5094b08ef02d" />





```
pd.concat([cc,new],axis=1)
```

<img width="576" alt="a27" src="https://github.com/user-attachments/assets/23442958-7834-4b2e-aa24-193a96448ace" />





# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file was successfully executed.


       
