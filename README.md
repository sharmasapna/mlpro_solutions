## mlpro solutions

This page contains solutions to some problems presented in [mlpro](https://mlpro.io/problems/) platform


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
