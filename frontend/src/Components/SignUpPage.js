import React, { useState } from 'react';
import axios from 'axios';
import './SignUpPage.css';
import HomePage from './Homepage.js';

const SignupPage = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('applicant');
  const [error, setError] = useState('');
  const [showHomePage, setShowHomePage] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/signup', {
        name,
        email,
        password,
        userType,
      });

      if (response.data.success) {
        // Handle successful signup
        console.log(response.data.message);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('An error occurred during signup');
    }
  };

  const handleGoBackToHomePage = () => {
    setShowHomePage(true);
  };

  if (showHomePage) {
    return <HomePage />;
  }

  return (
    <>
      <button type="button" className="goBack" onClick={handleGoBackToHomePage}>
        Go Back to Home Page
      </button>
      <div>
        <h2>Sign Up</h2>
        {error && <p>{error}</p>}
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="name">Name</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div>
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div>
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
          <button type="submit">Sign Up</button>
        </form>
      </div>
    </>
  );
};

export default SignupPage;
