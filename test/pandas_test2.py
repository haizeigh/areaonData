import pandas as pd
import numpy as np

testList = [4,4,5,3]
print(str(testList))

series = pd.Series(testList)
counts = series.value_counts().sort_index(axis=0, ascending=True)
print(str(counts))



df = pd.DataFrame({'A':np.array([1,np.nan,2,3,6,np.nan]),
                 'C':np.array([np.nan,4,np.nan,5,9,np.nan]),
                  'B':'foo'})
columns = df.columns
values = columns.sort_values()

values_data = df[values]
values_data.columns={'feature'}
print(values_data)

