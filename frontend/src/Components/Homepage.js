import React, { useState } from 'react';
import './HomePage.css';
import LoginPage from './LoginPage';
import SignupPage from './SignUpPage';

//import './LoginPage.css'
const HomePage = () => {
  const [currentPage, setCurrentPage] = useState('');

  const handleLoginClick = () => {
    setCurrentPage('login');
  };

  const handleSignUpClick = () => {
    setCurrentPage('signup');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'login':
        return <LoginPage />;
      case 'signup':
        return <SignupPage />;
      default:
        return (
          <div className="homepage-container">
            <h1 className="homepage-heading">Welcome to AI Recruiter</h1>
            <p>Please choose your action:</p>
            <div className="action-buttons">
              <button className="action-button login" onClick={handleLoginClick}>Login</button>
              <button className="action-button signup" onClick={handleSignUpClick}>Sign Up</button>
            </div>
          </div>
        );
    }
  };

  return (
    <div>
      {renderPage()}
    </div>
  );
};

export default HomePage;
