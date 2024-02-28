import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns
from plotly.offline import iplot , plot 

#importing original data 

jobs = pd.read_csv("jobs_in_data.csv")
print(jobs.head())
print(jobs.isnull().sum().sort_values())
print(jobs.info())
print(jobs["job_title"].value_counts())
print(jobs["job_category"].value_counts())
print(jobs["experience_level"].value_counts())
print(jobs["company_size"].value_counts())
print(jobs["employment_type"].value_counts())

print(jobs[jobs["company_location"]=="United States"].groupby(["job_title"]).agg(average_salaray =("salary", "mean"))[:10].round(2))
print(jobs[jobs["company_location"]=="United States"].groupby(["job_title", "experience_level"]).agg(average_salaray =("salary", "mean"))[:5].round(2))
data_engineer = jobs[jobs["job_title"]=="Data Engineer"].groupby(["company_location"]).agg(average_salary =("salary","mean"))[:10].round(2)
print(data_engineer.sort_values(by="average_salary", ascending=False))
data_scientist = jobs[jobs["job_title"]=="Data Scientist"].groupby(["company_location"]).agg(average_salary =("salary","mean"))[:10].round(2)
print(data_scientist.sort_values(by="average_salary", ascending=False))
data_analyst = jobs[jobs["job_title"]=="Data Analyst"].groupby(["company_location"]).agg(average_salary =("salary","mean"))[:10].round(2)
print(data_analyst.sort_values(by="average_salary", ascending=False))
machine_learning_engineer = jobs[jobs["job_title"]=="Machine Learning Engineer"].groupby(["company_location"]).agg(average_salary =("salary","mean"))[:10].round(2)
print(machine_learning_engineer.sort_values(by="average_salary", ascending=False))

data_engineer_usd = jobs[jobs["job_title"]=="Data Engineer"].groupby(["company_location"]).agg(average_salary =("salary_in_usd","mean"))[:10].round(2)
print(data_engineer_usd.sort_values(by="average_salary", ascending=False))
data_scientist_usd = jobs[jobs["job_title"]=="Data Scientist"].groupby(["company_location"]).agg(average_salary =("salary_in_usd","mean"))[:10].round(2)
print(data_scientist_usd.sort_values(by="average_salary", ascending=False))
data_analyst_usd = jobs[jobs["job_title"]=="Data Analyst"].groupby(["company_location"]).agg(average_salary =("salary_in_usd","mean"))[:10].round(2)
print(data_analyst_usd.sort_values(by="average_salary", ascending=False))
machine_learning_engineer_usd = jobs[jobs["job_title"]=="Machine Learning Engineer"].groupby(["company_location"]).agg(average_salary =("salary_in_usd","mean"))[:10].round(2)
print(machine_learning_engineer_usd.sort_values(by="average_salary", ascending=False))

data_analyst_max = jobs[jobs["job_title"]=="Data Analyst"].groupby(["company_location"]).agg(max_salary =("salary","max"))[:10].round(2)
print(data_analyst_max.sort_values(by="max_salary", ascending=False))
machine_learning_engineer_max = jobs[jobs["job_title"]=="Machine Learning Engineer"].groupby(["company_location"]).agg(max_salary =("salary","max"))[:10].round(2)
print(machine_learning_engineer_max.sort_values(by="max_salary", ascending=False))

print(jobs[jobs["company_location"]=="United States"].groupby("job_title")[["salary"]].max().sort_values(by="salary", ascending=False)[:10].round(2))
print(jobs[jobs["company_location"]=="United Kingdom"].groupby("job_title")[["salary"]].max().sort_values(by="salary", ascending=False)[:10].round(2))
print(jobs[jobs["company_location"]=="Turkey"].groupby("job_title")[["salary"]].max().sort_values(by="salary", ascending=False)[:10].round(2))


fig = px.histogram(jobs, x="job_title", y="salary", title="Job Title vs Salary", hover_data="company_location",
                   color="company_size", width=900, height=700)
fig.show()

fig = px.histogram(jobs, x="job_category", y="salary_in_usd", title="Job Title vs Salary", hover_data="work_year",
                   color="company_size", width=900, height=700)
fig.show()

country = ["United States", "Germany", "Canada", "United Kingdom", "France", "Australia", "Belgium", "Turkey", "Finland", "Brazil"]

#Top 15 Most Frequent Country
country = jobs["company_location"].value_counts()
top_10 = country.nlargest(10)
print(top_10)
top_10_df = pd.DataFrame({"Country":top_10.index, "Count":top_10.values})

plt.figure(figsize=(12,6))
ax = sns.barplot(data=top_10_df, x="Count", y="Country", palette="ocean")
plt.xlabel("Count")
plt.ylabel("Country")
plt.title("Top 10 Most Frequent Country")
plt.show()

#Top 15 Most Frequent Job Title
job = jobs["job_title"].value_counts()
job_10 = job.nlargest(10)
print(job_10)
job_10_df = pd.DataFrame({"Job Titles":job_10.index, "Count":job_10.values})

plt.figure(figsize=(8,6))
ax = sns.barplot(data=job_10_df, x="Job Titles", y="Count", palette="bright")
plt.xlabel("Job Titles")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Top 15 Most Frequent Job Titles")
plt.show()

job_title_USD = jobs.groupby('job_title')['salary_in_usd'].sum()
print(f"Top Year for Number of Employees '{job_title_USD.idxmax()}' with Salary '{job_title_USD.max()}'")
print(f"Least Year for Number of Employees '{job_title_USD.idxmin()}' with Salary '{job_title_USD.min()}'")
fig1 = (px.bar(job_title_USD[:10],
            labels={"job_title":"Job Titles",
                     "value": "Value"},
                 template='plotly_dark',
                 title="Top Job Titles in the World",
                 text_auto=True,
                 color_discrete_sequence=["#dd0be0"]))
fig1.show()


jobs_salary_in_usd = jobs.groupby(['work_year','job_title'])['salary_in_usd'].mean()
colors = ['#4a289b', '#a53d3d', '#268e7e', '#e60e0e']
i = 0
for j in range(2020, 2024):
    fig = px.bar(jobs_salary_in_usd.get(j)[:10],
           labels={"job_title":"Job Titles",
                   "value":"Average of Salary"},
            title=f"Average Salary of Jobs for Year in {j}",
            color_discrete_sequence=[colors[i]],
            template="plotly_dark",
            text_auto=True,
            orientation="h"       
                   )
    i+=1
fig.show()


employee_residence = jobs['employee_residence'].value_counts()
print(employee_residence)
fig2 = (px.bar(employee_residence[:10],
             labels={"employee_residence":"Name of Country",
                     "value": "Value"},
                    template='plotly_dark',
                    title="Top Countries in the World",
                    text_auto=True,
                    color_discrete_sequence=["#dd0be0"]))
fig2.show()

experience_level = jobs.groupby(["experience_level"])["salary"].mean()
fig4 = px.bar(experience_level[:10],
            labels={"experience_level":"Experience Level",
                    "salary": "Average of Salary"},
            template='plotly_dark',
            title="Top Experience Level Job Titles in the World",
            text_auto=True,
            color_discrete_sequence=["#dd0be0"])
fig4.show()

work_setting = jobs["work_setting"].value_counts()
fig5 = px.pie(values=work_setting.values,
              names=["In-person", "Remote", "Hybrid"],
              title="Type of Work Setting",
              color_discrete_sequence=["#dd0be1"])
fig5.show()



fig6 = px.line(x=jobs["work_year"].value_counts().index,
        y=jobs["work_year"].value_counts().values,
        markers=True,
        labels={'x':'Year','y':'Number of Employees'},
        title="Years of Work",
        line_shape="linear",
        color_discrete_sequence=['#cc2114'],
        template='plotly_dark'
        )
fig6.show()


#Correlation heatmap
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
jobs["exp_level"] = label_encoder.fit_transform(jobs["experience_level"])
jobs["company_size_encoded"] = label_encoder.fit_transform(jobs["company_size"])

corr_matrix = jobs[["salary_in_usd", "exp_level", "company_size_encoded"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="ocean", cbar=True)
plt.title("Correlation Matrix")
plt.show()


sns.countplot(data=jobs, x="work_year", hue="company_size", palette="icefire")
plt.xticks(fontsize=10,rotation=50)
plt.show()


plt.figure(figsize=(7,5))
sns.countplot(data=jobs, x="job_category", hue="work_year", palette="icefire")
plt.xticks(fontsize=8,rotation=50)
plt.show()

employee_residence_salary_top = jobs.groupby("employee_residence")["salary_in_usd"].mean().sort_values(ascending=False)[:5]
sns.barplot(x=employee_residence_salary_top.index, y=employee_residence_salary_top.values)
plt.title('Top 5 Average Salaries by Employee Residence')
plt.xticks(rotation=45)
plt.show()

employee_residence_salary_bottom = jobs.groupby("employee_residence")["salary_in_usd"].mean().sort_values(ascending=False).tail(5)
sns.barplot(x=employee_residence_salary_bottom.index, y=employee_residence_salary_bottom.values)
plt.title('Bottom 5 Average Salaries by Employee Residence')
plt.xticks(rotation=45)
plt.show()


salary_curr = jobs["salary_currency"].value_counts()
print(salary_curr)

fig7 = px.pie(values=salary_curr.values[:20],
              names=["USD", "GBP", "EUR", "CAD", "AUD", "PLN", "SGD", "CHF", "BRL", "TRY", "DKK"],
              title="Type of Salary Currency"
              )

fig7.show()


fig8= px.scatter(data_frame=jobs, x="experience_level", y="salary_in_usd", hover_data="job_title",color="job_category")
fig8.update_layout(title_text="Experience: The Ascending Path to Compensation", title_font=dict(size=16, family="Arial", color="darkblue"))
fig8.show()


#Model selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
jobs = pd.get_dummies(jobs)
X = jobs.drop(["salary"], axis=1)
y = jobs["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Reg
linear_reg = LinearRegression()
#fit the model training data
linear_reg.fit(X_train, y_train)

# Initialize Random Forest Regression model
rf = RandomForestRegressor(random_state=42)
# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
linear_reg_pred = linear_reg.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluate the models
linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("Mean Squared Error (Linear Regression):", linear_reg_mse)
print("Mean Squared Error (Random Forest):", rf_mse)
