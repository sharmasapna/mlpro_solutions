## mlpro solutions

This page contains solutions to some problems presented in [mlpro](https://mlpro.io/problems/) platform



### Data Wrangling - 1D Histogram
#### Difficulty: 2

A histogram represents the frequency distribution of data. The idea is to take a list of values and make a tally of how many times each value occurs.
Given a 1D NumPy array, create a histogram of the data represented as a NumPy array where the index represents the number and the value represents the count. Note that the first index (position 0) represents how many times number 1 occurs.

There are 2 things to note:

The parameter range default from min to max, however the task requires histogram starting from 1.
bins default to only 10 values, therefore we must set it to max cover all the numbers in the np.array
```
import numpy as np

def get_histogram(arr):
    h,b = np.histogram(arr,bins=np.amax(arr),range = (1,np.amax(arr)))
    return h
```
### Data Wrangling - Normalization of Data between 0 and 1
#### Difficulty: 2
When performing statistical analyses, data should be in a normalized form.   
Normalization is used to fit the data within unity so that all values must    
fall in range between 0 and 1.     
There are different kinds of normalization but here we will use Min-Max Normalization, which is:
x = (x-x_min)/(x_max-x_min)
Input Details: NumPy array of random floating point values.
Output Details: A NumPy array with data in normalized form ranging from 0 to 1.
```
def minMaxNormalization(inputArray):
    # calculate the minimum and the maximum value
    min_val = min(inputArray)
    max_val = max(inputArray)
    
    res = []
    for i in range(len(inputArray)):
        new_val = (inputArray[i] - min_val)/(max_val-min_val)
        res.append(new_val)
    return np.array(res)
 ```

### NLP -  Jaccard Similarity Index
#### Difficulty: 2
The Jaccard similarity index measures the similarity between two sets of data. It can range from 0 to 1. The higher the number, the more similar the two sets of data.   
The Jaccard similarity index is calculated as:   
Jaccard Similarity = (number of observations in both sets:**Intersection**) / (number in either set:**Union**)
```
def jaccard_similarity(list1,list2):
    number_of_elements_in_either_sets = len(list(set(list1).intersection(list2)))
    number_of_elements_in_both_sets   = len(list1) + len(list2) - number_of_elements_in_either_sets
    js = number_of_elements_in_either_sets / number_of_elements_in_both_sets 
    return js
```
### Computer Vision : Image Mirror
#### Difficulty: 5


Image mirroring or flipping is a data augmentation preprocessing technique used in computer vision to add some variance to the dataset and increase the number of training examples, helping to reduce overfitting.
In this problem, you are given an image in the form of 2-D matrix. Write a function that returns that image flipped or mirrored.
```
import numpy as np
def flip_img(image):
    res = []
    for val in image:
      res.append(val[::-1])
    return np.array(res)

```

### Computer Vision : Simple Blur
#### Difficulty: 1

In this problem, you are given a NumPy array (2D) representing a grayscale image. Your job is to read this image and blur it using the image processing library Pillow. The output should also be a NumPy array representing the blurred image.   
```
from PIL import Image, ImageFilter
import numpy as np

# Please do not change the below function name and parameters
#def blur_PIL(img_array):
def blur_PIL(img_array: np.ndarray):
    
    # converting the list to numpy array
    shape = img_array.shape
    im = np.asarray(img_array).reshape(-1,1)
    out = np.concatenate([im, np.ones(np.prod(shape)).reshape(-1,1) * 255], axis=1).reshape(shape+(2,))
    return out.astype(np.uint8)   
```
### Computer Vision - Alpha Blending
#### Difficulty: 3

Alpha blending is the process of overlaying a foreground image with transparency over a background image. The transparency is often the fourth channel of an image ( e.g. in a transparent PNG), but it can also be a separate image. This transparency mask is often called the alpha mask or the alpha matte.
```
import numpy as np
def alpha_blend(mat1, mat2):
    m1 = np.asarray(mat1)
    m2 = np.asarray(mat2)
    return np.floor(m1 + 0.2*(m2 - m1))
 ```
### Fundamentals : Outlier Detection with IQR
#### Difficulty: 2
IQR, or Inter-Quartile Range, is the difference between third quartile and first quartile of your data (75th percentile and 25th percentile respectively).
Lower and upper bounds are calculated as Q1 - 1.5 IQR and Q3 + 1.5 IQR respectively.
Write a function that takes a 1D numeric NumPy array and returns a 1D array consistting of the outliers that are greater than upper bound or less than lower bound calculated.


```
import numpy as np
def outliers_with_IQR(data):
    
    # calculating the first and third quaritle values
    q1,q3 = np.percentile(data,[25,75])
    
    # calculating the interquartile range
    iqr = q3 - q1
    
    # calculating the upper and lower limits
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr
    
    res = []
    # searching the outliers
    for val in data:
      if lower_limit > val:
        res.append(val)
      elif val > upper_limit:
          res.append(val)
    
    res = np.array(res)
    return res
```
### Feature Engineering
#### Difficulty: 7
Apply the following feature engineering to both X_train and X_test:
Add two columns of squares of the two features
Add two columns of log of the two features
Add two columns of exp of the two features
The final feature vector should consist of the concatenation of the original X_train with the squares, log, and exp features, in that order.
```
import numpy as np
from sklearn.linear_model import LogisticRegression
def Predict(X_train, Y_train, X_test):
    # Feature engineering
    def get_transformed_data(df):
      x = pd.DataFrame(df,columns=('a','b'))
      x['a_sqr'] = x.a*x.a
      x['b_sqr'] = x.b*x.b
      x['a_log'] = np.log(x.a)
      x['b_log'] = np.log(x.b)
      x['a_exp'] = np.exp(x.a)
      x['b_exp'] = np.exp(x.b)
      x = np.array(x)
      return x.tolist()
      
    X_train = get_transformed_data(X_train)
    X_test = get_transformed_data(X_test)
    # creating the model
    model = LogisticRegression(solver='liblinear').fit(X_train,Y_train)
    return model.predict(X_test)
```

### Prob. and Stat
#### Difficulty: 2

A bunch of students take an exam with mean score mu and standard deviation of stdev.   
What is the probability that a random student scored above a 95 on the exam?
```
import math
def normal_cdf(mu, stdev):
    # calculating the z_score
    z_score = (95-mu)/stdev
    
    # The math.erf() method returns the error function 
    # of a number.
    # This method accepts a value between - inf and + inf, 
    # and returns a value between - 1 to + 1.
    p = .5 * (math.erf(z_score / 2 ** .5) + 1)
    
    # since the value is the probability of a number
    # being in between 0 to 95, we need to subtract
    # it from 1 to get the value between 95 to 100
    return 1-p
```

### Prob. and Stat - Absolute Error
#### Difficulty: 2
```
import numpy as np
def abs_error(arr1,arr2):
    ae = 0
    for a,b in zip(arr1,arr2):
      ae += np.abs(a-b)
    return ae/len(arr1)
```


### Prob. and Stat - Cumulative Sum
#### Difficulty: 2

```
def csum(n):
    res = [n[0]]
    for i in range(1,len(n)):
        cs = res[i-1] + n[i]
        print(cs)
        res.append(cs)
    return res
```

### Prob. and Stat - Bayes Theorem
#### Difficulty: 1

```
import numpy as np

def bayes(h_prior, e_given_h,  e_not_given_h):
    
    # here we have to return p(H|E) while we are given 
    # p(H),P(E|H) and p(E|notH)(assuming it is e_not_given_h))
    # Applying Bayes theorm
    # (p(H|E)) = p(H)*p(E|H)/(p(H)*p(E|H)  + p(notH)*p(E|notH))
    return h_prior*e_given_h/(h_prior*e_given_h + (1-h_prior)*e_not_given_h)

```
