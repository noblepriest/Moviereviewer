import fastapi
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI,Response,status
from fastapi import FastAPI,Depends
import uvicorn
import spacy
from predict import parsedata
from predict import findemail
from predict import extract_website
from predict import extract_numbers
from transformers import BertTokenizer, BertForSequenceClassification
from special_case import detect


class Item(BaseModel):
    text: str = None



app = FastAPI(debug=True)

@app.on_event("startup")
async def startup_event():
    global model
    global tokenizer
    print("===========MODEL LOADING====================")
    model_path_1="IMDB_model_bert/"
    model = BertForSequenceClassification.from_pretrained(model_path_1)
    tokenizer= BertTokenizer.from_pretrained(model_path_1)
    print('===========MODEL LOADED=================')

@app.post("/firsttask",status_code=200)
async def classify(response: Response ,item:Item):
    result = detect(item.text,model,tokenizer)
    return result



@app.post("/secondtask",status_code=200)
async def infoextract(response: Response ,item:Item):
    #print(item.text)
    name = parsedata(item.text)
    email = findemail(item.text)
    website = extract_website(item.text)
    phone_number = extract_numbers(item.text)
    if name==None:
        name = "could not extract"
    else:
        pass

    extracted_data ={"Name":name,"email":email,"website":website,"Phone_number":phone_number}
    return extracted_data



if __name__ == "__main__":
    uvicorn.run("text_app:app", host="0.0.0.0", port=1997)