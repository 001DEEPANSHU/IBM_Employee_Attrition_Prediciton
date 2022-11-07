import pickle
import streamlit as st
import pandas as pd


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


Age = col31.number_input("Insert Age")   #7th feature
DailyRate = col32.number_input("Insert DailyRate") #8th feature
DistanceFromHome = col33.number_input("Insert DistanceFromHome") #9th feature

col41, col42, col43 = st.columns(3)

Education = col41.number_input("Insert Education")   #10th feature
EnvironmentSatisfaction = col42.number_input("Insert EnvironmentSatisfaction") #11th feature
HourlyRate = col43.number_input("Insert HourlyRate") #12th feature


col51, col52, col53 = st.columns(3)

JobInvolvement = col51.number_input("Insert JobInvolvement")   #13th feature
JobLevel = col52.number_input("Insert JobLevel") #14th feature
JobSatisfaction = col53.number_input("Insert JobSatisfaction") #15th feature


col61, col62, col63 = st.columns(3)

MonthlyIncome = col61.number_input("Insert MonthlyIncome")   #16th feature
MonthlyRate = col62.number_input("Insert MonthlyRate") #17th feature
NumCompaniesWorked = col63.number_input("Insert NumCompaniesWorked") #18th feature



col71, col72, col73 = st.columns(3)

PercentSalaryHike = col71.number_input("Insert PercentSalaryHike")   #19th feature
PerformanceRating = col72.number_input("Insert PerformanceRating") #20th feature
RelationshipSatisfaction = col73.number_input("Insert RelationshipSatisfaction") #21th feature


col81, col82, col83 = st.columns(3)

StockOptionLevel = col81.number_input("Insert StockOptionLevel")   #22th feature
TotalWorkingYears = col82.number_input("Insert TotalWorkingYears") #23th feature
TrainingTimesLastYear = col83.number_input("Insert TrainingTimesLastYear") #24th feature


col91, col92, col93 = st.columns(3)

WorkLifeBalance = col81.number_input("Insert WorkLifeBalance")   #25th feature
YearsAtCompany = col82.number_input("Insert YearsAtCompany") #26th feature
YearsInCurrentRole = col83.number_input("Insert YearsInCurrentRole") #27th feature



col101, col102, col103 = st.columns(3)

YearsSinceLastPromotion = col81.number_input("Insert YearsSinceLastPromotion")   #28th feature
YearsWithCurrManager = col82.number_input("Insert YearsWithCurrManager") #29th feature


input_data = [

{
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
]

input_df = pd.DataFrame(input_data)

pred = pipeline.predict_proba(input_df)[0,1]
pred = float(pred)
print(pred)

col104, col105 = st.columns(2)

st.write('The input data is') 
st.table(input_df)

col104.write("Probability of Employee's attrition is:")

col105.write(
    f"""<p class="big-font">
{pred:0.4f}
</p>
""",
    unsafe_allow_html=True,
)