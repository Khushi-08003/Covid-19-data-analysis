#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots 
from datetime import datetime


# In[8]:


covid_df = pd.read_csv("C:\\Users\\KHUSHI\\Downloads\\Covid-19-Data-Analysis\\covid_19_india.csv")


# In[9]:


covid_df.head(10)


# In[13]:


covid_df.info()


# In[15]:


covid_df.describe()


# In[17]:


vaccine_df = pd.read_csv("C:\\Users\\KHUSHI\\Downloads\\Covid-19-Data-Analysis\\covid_vaccine_statewise.csv")


# In[19]:


vaccine_df.head()


# In[21]:


covid_df.drop(["Sno","Time" , "ConfirmedIndianNational", "ConfirmedForeignNational"], inplace = True, axis = 1)


# In[23]:


covid_df.head()


# In[25]:


covid_df['Date'] = pd.to_datetime(covid_df['Date'], format = '%Y-%m-%d')


# In[27]:


covid_df.head()


# In[29]:


#Active cases

covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# In[31]:


statewise = pd.pivot_table(covid_df,values = ['Confirmed', 'Deaths', 'Cured'], index = 'State/UnionTerritory', aggfunc = 'max')


# In[33]:


statewise['Recovery Rate'] = statewise['Cured']*100/statewise['Confirmed']


# In[35]:


statewise['Mortality Rate'] = statewise['Deaths']*100/statewise['Confirmed']


# In[37]:


statewise = statewise.sort_values(by = 'Confirmed', ascending = False)


# In[39]:


statewise.style.background_gradient(cmap = 'cubehelix')


# In[41]:


#Top 10 active cases states

top_10_active_cases = covid_df.groupby(by = 'State/UnionTerritory').max()[['Active_Cases','Date']].sort_values(by = ['Active_Cases'],ascending = False).reset_index()


# In[43]:


fig = plt.figure(figsize=(16,9))


# In[45]:


plt.title('Top 10 states with most active cases in India', size = 25)


# In[47]:


ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = 'Active_Cases', x = 'State/UnionTerritory', linewidth = 2, edgecolor ='black')


# In[49]:


#Top 10 active cases states

top_10_active_cases = covid_df.groupby(by = 'State/UnionTerritory').max()[['Active_Cases','Date']].sort_values(by = ['Active_Cases'],ascending = False).reset_index()
fig = plt.figure(figsize=(16,9))
plt.title('Top 10 states with most active cases in India', size = 25)
ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = 'Active_Cases', x = 'State/UnionTerritory', linewidth = 2, edgecolor ='black')
plt.xlabel('States')
plt.ylabel('Total Active Cases')
plt.show()


# In[51]:


# Top states with highest deaths

top_10_deaths = covid_df.groupby(by = 'State/UnionTerritory').max()[['Deaths','Date']].sort_values(by = ['Deaths'], ascending = False).reset_index()

fig = plt.figure(figsize=(18,5))

plt.title('Top 10 states with most Deaths', size = 25)

ax = sns.barplot(data = top_10_deaths.iloc[:12], y = 'Deaths' , x = 'State/UnionTerritory',linewidth = 2, edgecolor = 'black')

plt.xlabel('States')
plt.ylabel('Total Death Cases')
plt.show()



# In[55]:


# Growth trend
#ERROR AARHA ISME 

fig = plt.figure(figsize = (12,6))

ax = sns.lineplot(data = covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra','Karnataka','Kerala','Tamil Nadu','Uttar Pradesh'])],
                  x = 'Date', y ='Active_Cases' , hue = 'State/UnionTerritory')

ax.set_title('Top 5 Affected States in India', size = 16)


# In[63]:


vaccine_df.head()


# In[65]:


vaccine_df.rename(columns = {'Updated On' : 'Vaccine_Date'}, inplace = True)


# In[67]:


vaccine_df.head(10)


# In[69]:


vaccine_df.info()


# In[71]:


vaccine_df.isnull().sum()


# In[73]:


vaccination = vaccine_df.drop(columns = ['Sputnik V (Doses Administered)','AEFI','18-44 Years (Doses Administered)','60+ Years (Doses Administered)'],axis=1)


# In[75]:


vaccination.head()


# In[77]:


# Male vs Female Vaccination 
male = vaccination['Male(Individuals Vaccinated)'].sum()
female = vaccination['Female(Individuals Vaccinated)'].sum()
px.pie(names=['Male','Female'], values=[male,female], title = 'Male and Female Vaccination',color_discrete_sequence=px.colors.sequential.RdBu)


# In[79]:


# Remove rows where state = India 

vaccine = vaccine_df[vaccine_df.State!='India']
vaccine


# In[81]:


vaccine.rename(columns ={'Total Individuals Vaccinated' : 'Total'}, inplace = True)
vaccine.head()


# In[83]:


# Most vaccinated State

max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total', ascending = False)[:5]
max_vac


# In[85]:


fig = plt.figure(figsize=(10,5))
plt.title('Top Vaccinated States in India', size = 20)
x = sns.barplot(data = max_vac.iloc[:10],y = max_vac.Total, x = max_vac.index, linewidth=2,edgecolor = 'black',palette='rocket')
plt.xlabel('States')
plt.ylabel('Vaccination')
plt.show()


# In[90]:


fig = plt.figure(figsize=(10, 5))

# Set the title for the plot
plt.title('Least Vaccinated States in India', size=20)

# Sort the DataFrame to get the least vaccinated states (smallest values at the top)
min_vac = max_vac.sort_values(by='Total').iloc[:10]  # Assuming 'Total' is the column for vaccination counts

# Create the bar plot
x = sns.barplot(
    data=min_vac,
    y='Total',  # Column with the vaccination totals
    x=min_vac.index,  # Assuming the index contains state names
    linewidth=2,
    edgecolor='black',
    palette='rocket'
)

# Set the labels for the x and y axes
plt.xlabel('States')
plt.ylabel('Vaccination')

