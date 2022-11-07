import pickle
import streamlit as st


def load_pipeline():
    model = './models/pipe.bin'

    with open(model, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    return pipeline


pipeline = load_pipeline()

col11, col12, col13 = st.columns(3)

BusinessTravel = col11.selectbox('Business Travel', ('Travel_Frequently', 'Travel_Rarely', 'Non-Travel')) #1st feature

Department = col12.selectbox('Department', ('Research & Development', 'Sales',  'Human Resources')) #2nd feature

EducationField = col13.selectbox(
    'Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other')
) #3rd feature




col21, col22, col23 = st.columns(3)
Gender = col21.selectbox('Gender', ('Female', 'Male')) #4th feature

JobRole = col22.selectbox(         #5th feature
    'Job Role',
    (
        'Sales Executive',
        'Research Scientist',
        'Laboratory Technician',
        'Manufacturing Director',
        'Healthcare Representative',
        'Manager',
        'Sales Representative',
        'Research Director',
        'Human Resorces',
    ),
)

MaritalStatus = col23.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))  #6th feature


col31, col32, col33 = st.columns(3)


Age = col31.slider(    #7th feature
    'Age',
    min_value=18,
    max_value=95,
    value=30,
)

DailyRate = col32.slider(  #8th feature
    'Daily Rate',
    min_value=0,
    max_value=2000,
    value=0,
)
DistanceFromHome = col33.slider(   #9th feature
    'Distance From Home',
    min_value=0,
    max_value=30,
    value=0,
)

col41, col42, col43 = st.columns(3)

Education = col41.slider(   #10th feature
    'Education',
    min_value=1,
    max_value=5,
    value=1,
)
EnvironmentSatisfaction = col42.slider(    #11th feature
    'Enviroment Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)

HourlyRate = col43.slider(      #12th feature
    'Hourly Rate',
    min_value=0,
    max_value=100,
    value=0,
)



col51, col52, col53 = st.columns(3)

JobInvolvement = col51.slider(    #13th feature
    'Job Involvement',
    min_value=1,
    max_value=4,
    value=1,
)
JobLevel = col52.slider(         #14th feature
    'Job Level',
    min_value=1,
    max_value=5,
    value=1,
)


JobSatisfaction = col53.slider(    #15th feature
    'Job Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)



col61, col62, col63 = st.columns(3)

MonthlyIncome = col61.slider(   #16th feature
    'Monthly Income',
    min_value=0,
    max_value=100000,
    value=0,
)
MonthlyRate = col62.slider(     #17th feature
    'Monthly Rate',
    min_value=0,
    max_value=30000,
    value=0,
)


NumCompaniesWorked = col63.slider(    #18th feature
    'Number of Companies Worked',
    min_value=0,
    max_value=10,
    value=0,
)



col71, col72, col73 = st.columns(3)

PercentSalaryHike = col71.slider(   #19th feature
    'Percent Salary Hike',
    min_value=0,
    max_value=30000,
    value=0,
)


PerformanceRating = col72.slider(    #20th feature
    'Performance Rating',
    min_value=1,
    max_value=4,
    value=1,
)

RelationshipSatisfaction = col73.slider(  #21st feature
    'Relationship Satisfaction',
    min_value=1,
    max_value=4,
    value=1,
)


col81, col82, col83 = st.columns(3)

StockOptionLevel = col81.slider(    #22nd feature
    'Stock Option Level',
    min_value=1,
    max_value=4,
    value=1,
)


TotalWorkingYears = col82.slider(   #23rd feature
    'Total Working Years',
    min_value=0,
    max_value=50,
    value=0,
)
TrainingTimesLastYear = col83.slider(   #24th feature
    'Training Times Last Year',
    min_value=0,
    max_value=10,
    value=0,
)



col91, col92, col93 = st.columns(3)

WorkLifeBalance = col91.slider(    #25th feature
    'Work Life Balance',
    min_value=1,
    max_value=4,
    value=1,
)


YearsAtCompany = col92.slider(   #26th feature
    'Years at Company',
    min_value=0,
    max_value=50,
    value=0,
)


YearsInCurrentRole = col93.slider(   #27th feature
    'Years in Current Role',
    min_value=0,
    max_value=50,
    value=0,
)


col101, col102, col103 = st.columns(3)

YearsSinceLastPromotion = col101.slider(    #28th feature
    'Years Since Last Promotion',
    min_value=0,
    max_value=50,
    value=0,
)

YearsWithCurrManager = col102.slider(   #29th feature
    'Years with Current Manager',
    min_value=0,
    max_value=50,
    value=0,
)


input_data = {
    'Age': int(Age),
    'BusinessTravel': BusinessTravel,
    'DailyRate': float(DailyRate),
    'Department': Department,
    'DistanceFromHome': float(DistanceFromHome),
    'Education': int(Education),
    'EducationField': EducationField,
    'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
    'Gender': Gender,
    'HourlyRate': float(HourlyRate),
    'JobInvolvement': int(JobInvolvement),
    'JobLevel': int(JobLevel),
    'JobRole': JobRole,
    'JobSatisfaction': int(JobSatisfaction),
    'MaritalStatus': MaritalStatus,
    'MonthlyIncome': float(MonthlyIncome),
    'MonthlyRate': float(MonthlyRate),
    'NumCompaniesWorked': int(NumCompaniesWorked),
    'PercentSalaryHike': float(PercentSalaryHike),
    'PerformanceRating': int(PerformanceRating),   
    'RelationshipSatisfaction': int(RelationshipSatisfaction),
    'StockOptionLevel': int(StockOptionLevel),
    'TotalWorkingYears': int(TotalWorkingYears),
    'TrainingTimesLastYear': int(TrainingTimesLastYear),
    'WorkLifeBalance': int(WorkLifeBalance),
    'YearsAtCompany': int(YearsAtCompany),
    'YearsInCurrentRole': int(YearsInCurrentRole),
    'YearsSinceLastPromotion': int(YearsSinceLastPromotion),
    'YearsWithCurrManager': int(YearsWithCurrManager),
}
