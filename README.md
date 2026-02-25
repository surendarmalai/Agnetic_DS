This repository contains the source code for the creation of an autonomous Data Science workflow for telecom industry
The agent workflow in mind for now is as follows

 - Agent 1: Field renaming - LLM based and renames the raw fields of the dataframe that has been initialised through sql. Gives the code for the same
 - Agent 2: Field type correction - works through the dataframe to correct the field types
 - Agent 3: Cleaning - Correct the categorical fields to have standardized categories and for numeric fields to have only numbers. Without the categories or numbers, None needs to be the default value
 - Agent 4: Filling - Appropriate action needs to be taken for imputing data into fields. for numeric, this will simply be filling with 0, but for categorical, other methods may have to be followed. Still in architectural phase
