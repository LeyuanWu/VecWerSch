import pandas as pd
from IPython.display import display

df1 = pd.DataFrame({'A': [1]}).T
df1.index.name = 'Reference'

df2 = pd.DataFrame({'A': [2]}).T
df2.index.name = 'Polyhedron'

df3 = df2 - df1
df3.index.name = 'Difference'

# 方法：分别创建 Styler
s1 = df1.style.format('{:.1e}').set_properties(**{'text-align': 'center'})
s2 = df2.style.format('{:.1e}').set_properties(**{'text-align': 'center'})
s3 = df3.style.format('{:.1e}').set_properties(**{'text-align': 'center'})

display(s1)
display(s2)
display(s3)