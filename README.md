# *z*-anonymity
This project contains the code used in the paper **_z_-anonymity: Zero-Delay Anonymization for Data Streams** currently under evaluation at ACM Performance Evaluation.

*z*-anonymity is a privacy property defined in a data stream scenario, where incoming tuples *(t, u, a)* describe that at time *t*, a user *u* has exposed an attribute *a*. z-anonymity is satisfied when no z-private attributes are released in a past time period &Delta;*t*.

The model defines a set of probabilities, given as input the threshold *z*, the time period &Delta;*t*, the number of users in the system *U*, the number of attributes *A*. Moreover, it is possible to input three alternative quantities:
- the exposing rate in a unit of time of the most popular attribute, from which the other attributes' rates will follow as 1/r, where r indicates the r-th most popular attributes;
- a list of the exposing rates in &Delta;*t*, one per attribute;
- a list of the exposing probabilities in &Delta;*t*, one per attibute.

Every quanitity in the previous list depends on the previous quantity. Whichever the input, the model will evaluate the needed probabilities, which include:
- the probability of an attribute *a* of being published in &Delta;*t*, when exposed;
- the probability of an attribute *a* of being published in &Delta;*t*
- the probability of a user in the system to be *k*-anonymized.

Other parameters are available to better understand the internal functioning of the model. Please refer to the class documentation.

The `zanon_model.py` contains the z-anonymity class, while the file `zanonymity.ipynb` shows which types of information it is possible to extract from an instantiation of the model. The `zanon_model_classes.py` contains the z-anonymity class designed to cope with two classes of users with different exposing rates.
