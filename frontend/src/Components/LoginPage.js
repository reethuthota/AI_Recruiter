import React, { useState } from 'react';
import axios from 'axios';
import './LoginPage.css'; // Import CSS file here
import RecruiterBase from './Recruiter/RecruiterBase.js'; 
import ApplicantBase from './Applicant/ApplicantBase.js';
import HomePage from './Homepage.js';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('applicant');
  const [error, setError] = useState('');
  const [loggedIn, setLoggedIn] = useState(false); // State to manage login status
  const [showHomePage, setShowHomePage] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/login', {
        email,
        password,
        userType,
      });
  
      if (response.data.success) {
        // Handle successful login
        console.log(response.data);
        const userType = response.data.user.userType;
        if (userType === 'recruiter') {
          // Set loggedIn to true if the user is a recruiter
          setLoggedIn(true);
          setUserType('recruiter'); // Set userType explicitly for recruiter
        } else if (userType === 'applicant') {
          // Set loggedIn to true if the user is an applicant
          setLoggedIn(true);
          setUserType('applicant'); 
          // Set userType explicitly for applicant
          console.log("loggedIn:", loggedIn);
          console.log("userType:", userType);
        } else {
          // Handle other user types
        }
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('An error occurred during login');
    }
  };

  const handleGoBackToHomePage = () => {
    setShowHomePage(true);
  };

  if (showHomePage) {
    return <HomePage />;
  }

  // Render RecruiterBase component if loggedIn is true
  if (loggedIn && userType==='recruiter') {
    return <RecruiterBase />;
  }
  if(loggedIn && userType ==='applicant'){
    return <ApplicantBase/>
  }

  return (
    <>
      <button type="button" className="goBack" onClick={handleGoBackToHomePage}>
        Go Back to Home Page
      </button>
      <div className="mainContainer">
        <h2>Login</h2>
        {error && <p>{error}</p>}
        <form onSubmit={handleSubmit}>
          <div className="inputEmail">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="inputBox"
            />
          </div>
          <div className="inputPassword">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="inputBox"
            />
          </div>
          <div className="inputContainer">
            <label htmlFor="userType">User Type</label>
            <select
              id="userType"
              value={userType}
              onChange={(e) => setUserType(e.target.value)}
            >
              <option value="applicant">Applicant</option>
              <option value="recruiter">Recruiter</option>
            </select>
          </div>
          <button type="submit" className="inputButton">Login</button>
        </form>
      </div>
    </>
  );
};

export default LoginPage;
