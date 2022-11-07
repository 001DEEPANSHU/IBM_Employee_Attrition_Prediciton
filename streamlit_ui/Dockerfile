FROM  python:3.9.12

RUN mkdir app models

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt


CMD streamlit run app/app.py