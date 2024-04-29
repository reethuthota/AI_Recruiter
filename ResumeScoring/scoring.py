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
    print(skills)
    return skills


def skills_matching(resume_skills, skills_df):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    
    primary_skills = set(skills_df['Primary'].str.split(',').explode().str.strip())
    secondary_skills = set(skills_df['Secondary'].str.split(',').explode().str.strip())
    resume_skills_list = resume_skills.split(',')

    primary_similarity_scores = []
    secondary_similarity_scores = []
    
    # Calculate similarity scores for primary skills
    prompt1 = f"Find the intersection between set 1 : {', '.join(resume_skills_list)} and set 2 : {primary_skills}. Give me a percentage as (intersection/number of items in set 2)*100. Print only the final percentage"
    docs_primary_skills = []
    for skill in primary_skills:
        docs_primary_skills.append(Document(page_content=skill))
    primary_similarity_scores = chain.run(input_documents=docs_primary_skills, question=prompt1, model='gpt-3.5-turbo')
    print("Primary Skills Match :" , primary_similarity_scores)
    

    # Calculate similarity scores for secondary skills
    prompt = f"Find the intersection between set 1 : {', '.join(resume_skills_list)} and set 2 : {secondary_skills}. Give me a percentage as (intersection/number of items in set 2)*100. Print only the final percentage"
    docs_secondary_skills = []
    for skill in secondary_skills:
        docs_secondary_skills.append(Document(page_content=skill))
    secondary_similarity_scores = chain.run(input_documents=docs_secondary_skills, question=prompt, model='gpt-3.5-turbo')
    print("Secondary Skills Match :" , secondary_similarity_scores)


resume_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/sample/Reethu_Resume.pdf'
resume_skills = extract_resume_info(resume_path)

skills_df = pd.read_csv('/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/skills.csv')
skills_matching(resume_skills, skills_df)
