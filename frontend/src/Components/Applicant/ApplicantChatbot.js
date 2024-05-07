import React, { useState, useEffect } from 'react';
import './ApplicantChatbot.css'; // Import CSS file for styling
import axios from 'axios';

function ApplicantChatbot() {
  const [step, setStep] = useState(0);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [file, setFile] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState('');
  
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/jobs');
        setJobs(response.data);
      } catch (error) {
        console.error('Error fetching jobs:', error);
      }
    };

    fetchJobs();
  }, []); // Empty dependency array ensures this effect runs only once when component mounts

  const handleNameChange = (event) => {
    setName(event.target.value);
  };

  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleJobSelect = (event) => {
    setSelectedJob(event.target.value);
  };

  const handleConfirm = async () => {
    if (step < 3) {
      setStep(step + 1);
      return;
    }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('email', email);
    formData.append('file', file);

    try {
      const response = await axios.post(`http://127.0.0.1:8000/api/apply/${selectedJob}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('Data uploaded successfully:', response.data);
    } catch (error) {
      console.error('Error uploading data:', error);
    }

    setStep(step + 1);
  };

  const handleRetry = () => {
    setName('');
    setEmail('');
    setFile(null);
    setSelectedJob('');
    setStep(0);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    handleConfirm();
  };

  return (
    <div className="chatbot-container">
      <h1>Chatbot</h1>
      <div className="chatbot-messages">
        {step === 0 && (
          <div className="chatbot-message">
            <p>Hi there! What's your name?</p>
            <form onSubmit={handleSubmit}>
              <input type="text" value={name} onChange={handleNameChange} required autoFocus />
              <button type="submit">Next</button>
            </form>
          </div>
        )}
        {step === 1 && (
          <div className="chatbot-message">
            <p>Thanks, {name}! What's your email?</p>
            <form onSubmit={handleSubmit}>
              <input type="email" value={email} onChange={handleEmailChange} required />
              <button type="submit">Next</button>
            </form>
          </div>
        )}
        {step === 2 && (
          <div className="chatbot-message">
            <p>Great, {name}! Please upload your PDF file.</p>
            <form onSubmit={handleSubmit}>
              <input type="file" accept=".pdf" onChange={handleFileChange} required />
              <button type="submit">Submit</button>
            </form>
          </div>
        )}
        {step === 3 && (
          <div className="confirmation-message">
            <p>Here's the information you provided:</p>
            <p>Name: {name}</p>
            <p>Email: {email}</p>
            <p>File: {file && file.name}</p>
            <p>Job Title: {selectedJob}</p>
            <p>Is this correct?</p>
            <button onClick={handleConfirm}>Yes, proceed</button>
            <button onClick={handleRetry}>No, retry</button>
          </div>
        )}
        {step === 4 && (
          <div className="success-message">
            <p>Your application has been submitted successfully!</p>
          </div>
        )}
      </div>

      {/* Job Selection Dropdown */}
      {step === 0 && (
        <div className="job-selection">
          <label>Select a Job:</label>
          <select value={selectedJob} onChange={handleJobSelect}>
            <option value="">Select</option>
            {jobs.map(job => (
              <option key={job.table_name} value={job.table_name}>{job.jobTitle}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

export default ApplicantChatbot;
