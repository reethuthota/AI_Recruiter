import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from PyPDF2 import PdfReader
import pandas as pd

from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

def extract_resume_info(resume_path):
    # Create a PdfReader object
    text = " "
    docs = []
    pdf_reader = PdfReader(resume_path)
    for page in pdf_reader.pages:
        text += page.extract_text() 
        docs.append(Document(page_content=text))
    
    prompt_template = """
        Extract the following information from the resume:
        
        Name: {}
        Email: {}
        Contact Info: {}
        Education: {}
        Skills: {}
        Experience: {}
        Designation: {}
    """
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    response = chain.run(input_documents=docs, question=prompt_template)
    
    parts = response.split('\n')
    
    extracted_info = {}
    for part in parts:
        part = part.strip()
        if part:
            key_value = part.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip() 
                value = key_value[1].strip()  
                extracted_info[key] = value

    df = pd.DataFrame([extracted_info])
    print(df)
    
    skills = extracted_info.get('Skills', '')
    
    # name = extracted_info.get('Name', '')
    # email = extracted_info.get('Email', '')
    # contact_info = extracted_info.get('Contact Info', '')
    # education = extracted_info.get('Education', '')
    # skills = extracted_info.get('Skills', '')
    # experience = extracted_info.get('Experience', '')
    # designation = extracted_info.get('Designation', '')
    
    # # Print the extracted information
    # print("Name:", name)
    # print("Email:", email)
    # print("Contact Info:", contact_info)
    # print("Education:", education)
    # print("Skills:", skills)
    # print("Experience:", experience)
    # print("Designation:", designation)


resume_path = '/Users/reethu/Documents/Job Docs/Resume final.pdf'
extract_resume_info(resume_path)
