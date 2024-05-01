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
    return response, skills


def skills_matching(resume_skills, skills_df):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    
    primary_skills = set(skills_df['Primary'].str.split(',').explode().str.strip())
    secondary_skills = set(skills_df['Secondary'].str.split(',').explode().str.strip())
    resume_skills_list = resume_skills.split(',')

    primary_similarity_scores = []
    secondary_similarity_scores = []
    
    # Calculate similarity scores for primary skills
    prompt1 = f"Find the intersection between set 1 : {', '.join(resume_skills_list)} and set 2 : {primary_skills}. Give me a percentage as (intersection/number of items in set 2)*100. Output only one final percentage and nothing extra. If you do not know, output 0."
    docs_primary_skills = []
    for skill in primary_skills:
        docs_primary_skills.append(Document(page_content=skill))
    primary_similarity_scores = chain.run(input_documents=docs_primary_skills, question=prompt1, model='gpt-3.5-turbo', temperature=0.7, max_tokens=3)
    print("Primary Skills Match :" , primary_similarity_scores)
    

    # Calculate similarity scores for secondary skills
    prompt = f"Find the intersection between set 1 : {', '.join(resume_skills_list)} and set 2 : {primary_skills}. Give me a percentage as (intersection/number of items in set 2)*100. Output only one final percentage and nothing extra. If you do not know, output 0."
    docs_secondary_skills = []
    for skill in secondary_skills:
        docs_secondary_skills.append(Document(page_content=skill))
    secondary_similarity_scores = chain.run(input_documents=docs_secondary_skills, question=prompt, model='gpt-3.5-turbo', temperature=0.7, max_tokens=3)
    print("Secondary Skills Match :" , secondary_similarity_scores)
    
def jobDescription_matching(resume_response, job_description_path):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    
    text = " "
    docs_jobDescription = []
    pdf_reader = PdfReader(job_description_path)
    for page in pdf_reader.pages:
        text += page.extract_text() 
    docs_jobDescription.append(Document(page_content=text))
    
    
    prompt = f"Give the job fit as a percentage for above job description and the given resume : {resume_response}.  Output only the percentage and omit any extra words. "
    jobDescription_matching_score = chain.run(input_documents=docs_jobDescription, question=prompt, model='gpt-3.5-turbo', temperature=0.7, max_tokens=3)
    print("Job Description Matching Score :" , jobDescription_matching_score)
    
    # pattern = r'\b(\d{1,3})%\b'
    # match = re.search(pattern, jobDescription_matching_score)
    # if match:
    #     percentage = match.group(1)
    #     print("Job Description Matching Final Score :" , percentage)


resume_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/sample/Reethu_Resume.pdf'
response, resume_skills = extract_resume_info(resume_path)


skills_df = pd.read_csv('/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/skills.csv')
skills_matching(resume_skills, skills_df)

job_description_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/Job_Description.pdf'
jobDescription_matching(response, job_description_path)