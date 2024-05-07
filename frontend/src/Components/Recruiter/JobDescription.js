import React, { useState } from 'react';
import axios from 'axios';
import './JobDescription.css'; // Import CSS file for styling

function Chatbot({ recruiterEmail }) {
  const [step, setStep] = useState(0);
  const [companyName, setCompanyName] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [email, setEmail] = useState('');
  const [file, setFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null); // State for CSV file
  const [uploadSuccess, setUploadSuccess] = useState(false); // State to track upload success

  const handleCompanyNameChange = (event) => {
    setCompanyName(event.target.value);
  };
  
  const handleJobTitleChange = (event) => {
    setJobTitle(event.target.value);
  };

  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleCsvFileChange = (event) => { // Function to handle CSV file change
    setCsvFile(event.target.files[0]);
  };

  const handleNextStep = () => {
    if (step === 0 && companyName.trim() !== '') {
      setStep(step + 1);
    } else if (step === 1 && jobTitle.trim() !== '') {
      setStep(step + 1);
    } else if (step === 2 && email.trim() !== '') {
      setStep(step + 1);
    } else if (step === 3 && file !== null) {
      setStep(step + 1);
    } else if (step === 4 && csvFile !== null) { // Check for CSV file
      handleSubmit();
    } else {
      alert('Please provide company name, job title, email, upload the PDF file, and CSV file.');
    }
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('recruiterEmail', email);
    formData.append('companyName', companyName);
    formData.append('jobTitle', jobTitle);
    formData.append('email', email);
    formData.append('file', file);
    formData.append('csvFile', csvFile); // Append CSV file to the form data

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/uploadJobDescription', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log(response.status);
      // Reset state and show thank you message
      setUploadSuccess(true); // Set upload success to true
      setCompanyName('');
      setJobTitle('');
      setEmail('');
      setFile(null);
      setCsvFile(null); // Reset CSV file state
      setStep(0); // Reset to initial step after successful upload
    } catch (error) {
      console.error('Error uploading job description:', error);
    }
  };

  const handleGoBack = () => {
    setUploadSuccess(false); // Reset upload success state
  };

  return (
    <div className="chatbot-container">
      <h1>Upload Job Description</h1>
      <div className="chatbot-messages">
        {step === 0 && (
          <div className="chatbot-message">
            <p>Hi there! What's your company's name?</p>
            <form>
              <input type="text" value={companyName} onChange={handleCompanyNameChange} required autoFocus />
              <button type="button" onClick={handleNextStep}>Next</button>
            </form>
          </div>
        )}
        {step === 1 && (
          <div className="chatbot-message">
            <p>Great, {companyName}! What's the title of the job description?</p>
            <form>
              <input type="text" value={jobTitle} onChange={handleJobTitleChange} required />
              <button type="button" onClick={handleNextStep}>Next</button>
            </form>
          </div>
        )}
        {step === 2 && (
          <div className="chatbot-message">
            <p>Thanks! Please provide your email.</p>
            <form>
              <input type="email" value={email} onChange={handleEmailChange} required />
              <button type="button" onClick={handleNextStep}>Next</button>
            </form>
          </div>
        )}
        {step === 3 && (
          <div className="chatbot-message">
            <p>Awesome, {jobTitle}! Now please upload the PDF file for the job description.</p>
            <form>
              <input type="file" accept=".pdf" onChange={handleFileChange} required />
              <button type="button" onClick={handleNextStep}>Next</button>
            </form>
          </div>
        )}
        {step === 4 && ( // New step for CSV file upload
          <div className="chatbot-message">
            <p>Great! Now please upload the Skills.txt file.</p>
            <form>
              <input type="file" accept=".txt" onChange={handleCsvFileChange} required />
              <button type="button" onClick={handleNextStep}>Submit</button>
            </form>
          </div>
        )}
        {uploadSuccess && ( // Display navigation button if upload success
          <div className="navigation-button">
            <button onClick={handleGoBack}>Go Back</button>
          </div>
        )}
        {step === 5 && (
          <div className="confirmation-message">
            <p>Thank you! Your job description and CSV file have been uploaded.</p>
            {uploadSuccess && (
              <div className="navigation-button">
                <button onClick={handleGoBack}>Go Back</button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Chatbot;