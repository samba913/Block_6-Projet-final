FROM python

WORKDIR /home/app

RUN pip install transformers boto3 pandas streamlit sklearn torch uvicorn
COPY . /home/app

CMD streamlit run --server.port 4001 app.py