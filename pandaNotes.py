# Create data
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35], 
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9], 
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

# Create data frame
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'deaths', 'battles', 'size', 'veterans', 'readiness', 'armored', 'deserters', 'origin'])
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])

xls_file = pd.ExcelFile('../data/example.xls')
xls_file

#display dataframe row in one line
pd.set_option('expand_frame_repr', False)

pd.set_option('expand_frame_repr', True)

# Set index
df = df.set_index('origin')

# View first few rows
df.head()

#view column
df['size']

#view multiple columns
df[['size', 'veterans']]

# Select all rows with the index label "Arizona"
df.loc[:'Arizona']

#add new row where all entries is bears
df.loc['Animal'] = 'Bears'

# Select every row up to 3
df.iloc[:2]

# Select the second and third row
df.iloc[1:3]

# Select every row beginning with the third row
df.iloc[2:]

# Select the first 2 columns
df.iloc[:,:2]

# Select rows where df.deaths is greater than 50
df[df['deaths'] > 50]

# Select rows where df.deaths is greater than 500 or less than 50
df[(df['deaths'] > 500) | (df['deaths'] < 50)]

# Select all the regiments not named "Dragoons"
df[~(df['regiment'] == 'Dragoons')]

# drop rows where certain column is NaN
df.dropna(subset = ['column_name'])

#rename column
df.rename(columns={'deaths': 'DEATHS'}, inplace=True)

#Sort the dataframes rows by reports, in descending order
df.sort_values(by='reports', ascending=0)

#Sort the dataframes rows by coverage and then by reports, in ascending order
df.sort_values(by=['coverage', 'reports'])

#boolean masking - will return True if value in column = Nighthawks
df['regiment'] == 'Nighthawks'

lambda inputs : expression
lambda a, b : a + b

s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])

#map lets you replace values based on corresponding dictionary key 
titles = {88 : "Director",
        3912  :  "Company Secretary",
        19682 :  "Non-designated Limited Liability Partnership Member",
        19683 :  "Designated Limited Liability Partnership Member",
        19681 :  "Judicial Factor",
        19907 :  "Manager appointed under the UK Charities Act",
        19906 :  "Manager appointed under the CAICE Act"}

df["Title2"] = df["Title2"].map(titles)

#add categorical ordering to series
s.astype('category',categories=['Low', 'Medium', 'High'],
                             ordered=True)

#create categories for series
s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])

pd.cut(s, 3)

# You can also add labels for the sizes [Small < Medium < Large].
pd.cut(s, 3, labels=['Small', 'Medium', 'Large'])

#trim dataframe with only certain columns
columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]

#display part of dataframe that meet a certain condition
df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]
only_gold = df.where(df['Gold'] > 0)

# assignment 3
import pandas as pd
# load excel file
xls_file = pd.ExcelFile('/Users/eagner/Downloads/ML/Energy Indicators.xls')
#see sheet names of excel file 
xls_file.sheet_names
[u'Energy']

df = xls_file.parse('Energy')


energy = pd.read_excel('/Users/eagner/Downloads/ML/Energy Indicators.xls', sheetname='Energy',names = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'],usecols=[2,3,4,5],skiprows=17,skip_footer=37,na_values="...")


#replace old values in country column with new values
energy = energy.replace({'Country' : 
    {"Republic of Korea" : "South Korea", 
    "United States of America20" : "United States", 
    "United Kingdom of Great Britain and Northern Ireland19" : "United Kingdom",
    "China, Hong Kong Special Administrative Region" : "Hong Kong",
    "Bolivia (Plurinational State of)" : "Bolivia",
    "Falkland Islands (Malvinas)" : "Falkland Islands",
    "Venezuela (Bolivarian Republic of)" : "Venezuela",
    "Sint Maarten (Dutch part)" : "Sint Maarten",
    "Iran (Islamic Republic of)" : "Iran",
    "Micronesia (Federated States of)" : "Micronesia"
    }})

#search column for string
energy[energy['Country'].str.contains("\)",na = False)]
#search column for numbers
energy[energy['Country'].str.contains("[0-9]",na = False)]

#tear out numbers in country column with nothing
energy.Country = energy.Country.str.replace("[0-9]",'').str.strip()

energy = energy.set_index('Country')

#multiply new field
energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x: x*1000000)

pd.read_csv('world_bank.csv')
gdp = pd.read_csv('/Users/eagner/Downloads/ML/world_bank.csv',index_col=0,skiprows=3)

gdp.iloc[:,:1]

#rename multiple indices
gdp = gdp.rename(index=
{"Korea, Rep.": "South Korea", 
"Iran, Islamic Rep.": "Iran",
"Hong Kong SAR, China": "Hong Kong"})

#put index values into a list
gdp.index.tolist()

gdp.loc["Korea, Rep."]

ScimEn = pd.read_excel('/Users/eagner/Downloads/ML/scimagojr.xlsx')

#how to do different types of joins between 2 dataframes
pd.merge(energy, gdp, how='outer', left_index=True, right_index=True)
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)

pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')

ScimEn = ScimEn.set_index('Country')

 ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

gdpenergy = pd.merge(energy, gdp[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']], how='inner', left_index=True, right_index=True)

gdpenergysci = pd.merge(ScimEn, gdpenergy, how='inner', left_index=True, right_index=True)

#question 1
top15 = gdpenergysci.sort_values(by='Rank').head(15)

#question 3 - calculate the average of rows across certain columns and sort in descending order
avgGDP = top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1).sort_values(ascending=False)

#question 4
top15.loc['France']['2015'] - top15.loc['France']['2006']

#question 5 - calculate the average of a column
top15['Energy Supply per Capita'].mean()

#question 6 - retrieve index of the maximum of a column
(top15['% Renewable'].idxmax(),top15['% Renewable'].max())

#question 7 - add new calculated values column
top15['Citation Ratio'] = top15['Self-citations']/top15['Citations']
(top15['Citation Ratio'].idxmax(),top15['Citation Ratio'].max())

#question 8 
top15['PopulationEstimate'] = top15['Energy Supply']/top15['Energy Supply per Capita']
#get third higest population and return name
top15.sort_values(by='PopulationEstimate', ascending=False).iloc[2].name

#question 9

top15['Citable docs per Capita'] = top15['Citable documents'] / top15['PopulationEstimate']
top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


import matplotlib as plt

#boolean masking cast as ints 
#question 10
top15['HighRenew'] = (top15['% Renewable'] > top15['% Renewable'].median()).astype(int)

#question 11
#to help with groupbys
#https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm
#https://chrisalbon.com/python/pandas_apply_operations_to_groups.html
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}

list(top15.groupby(ContinentDict))

groups = pd.DataFrame(columns = ['size', 'sum', 'mean', 'std'])
for group, frame in top15.groupby(ContinentDict):
    groups.loc[group] = [len(frame), frame['PopulationEstimate'].sum(),frame['PopulationEstimate'].mean(),frame['PopulationEstimate'].std()]

for group, frame in top15.groupby(ContinentDict):
    print type(group),frame

#question 11

top15 = top15.reset_index()
top15.rename(columns={'index': 'Country'}, inplace=True)
top15['Continent'] = [ContinentDict[country] for country in top15['Country']]
top15.groupby(['Continent','bins']).size()

chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

#standard deviation
distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))




