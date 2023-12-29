#import nltkS
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

#nltk.download('punkt')


app = FastAPI()


class TextRequest(BaseModel):
    text: str


class PredictionFeatures(BaseModel):
    textSize : int


model_name = "facebook/bart-large-cnn"

#loaded_model = tf.keras.models.load_model("model.h5")
#tokenizer = BartTokenizer.from_pretrained(model_name)
#model = BartForConditionalGeneration.from_pretrained(model_name)


def summarize_text(text_inp):
  
    # Load BART model and tokenizer
    loaded_model = tf.keras.models.load_model("model.h5")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    #Preprocess the text
    inputs = tokenizer.batch_encode_plus(         [text_inp],         max_length=1024,         truncation=True,         padding="longest",         return_tensors="pt"     )
    #   Generate the summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=150,
        early_stopping=True     )
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary

@app.post('/predict')
async def summarize_text(request: TextRequest):
    # Obtenez le texte à résumer à partir de la requête
    text = request.text

    # Utilisez Gensim pour générer le résumé du texte
    summary = summarize_text(text)

    # Retournez le résumé généré en tant que réponse JSON
    return {'predict': summary.tolist()[0]}




# #@app.post("/predict", tags=["Bart"])
# #async def predict(predictionFeatures: PredictionFeatures):
#     """
#     Prediction of salary for a given year of experience! 
#     """
#     # Read data 
#     text_size = pd.DataFrame({"textSize": [predictionFeatures.textSize]})

#     # Log model from mlflow 
#     #logged_model = 'runs:/323c3b4a6a6242b7837681bd5c539b27/salary_estimator'

#     # Load model as a PyFuncModel.
#     #loaded_model = mlflow.pyfunc.load_model(logged_model)
#     prediction = loaded_model.predict(text_size)

#     # Format response
#     response = {"prediction": prediction.tolist()[0]}
#     return response



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=4002)