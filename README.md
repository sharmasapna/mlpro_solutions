## mlpro_solutions

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
