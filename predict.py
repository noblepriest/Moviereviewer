import re
import spacy
def parsedata(data):
    
    # process the text with the model

    nlp = spacy.load('xx_ent_wiki_sm')
    doc = nlp(data)
    # extract Indian names
    for ent in doc.ents:
        if ent.label_ == 'PER': # check if the entity is a person name
            return ent.label_
        else:
            return "could not extract"
    

def findemail(data):
    email_addresses = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', data)
    if len(email_addresses)>0:
        return email_addresses[0]
    else:
        return "could not extract"


def extract_website(data):
    website_links = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www.*.com', data)
    if len(website_links)>0:
        return website_links[0]
    else:
        return "could not extract"


    
def extract_numbers(data):
    phone_numbers = re.findall(r'\b[789]\d{9}\b', data)
    if len(phone_numbers)>0:
        return phone_numbers[0]
    else:
        return "could not extract"



