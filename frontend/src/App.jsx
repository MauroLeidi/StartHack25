import './App.css'
import logo from './assets/logo.png'
import startGlobalLogo from './assets/start-global-logo.png'
import helblingLogo from './assets/helbling.png'

function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-brand">
          <img src={logo} alt="Logo" className="nav-logo" />
          <div className="brand-text">
            <h1 className="app-name">ADAPTIS</h1>
            <div className="tech-tagline">Working With You to Deliver More.</div>
          </div>
        </div>
      </nav>
      <div className="iframe-container">
        <div className="iframe-wrapper">
          <div className="iframe-outer-wrapper">
            <div className="iframe-inner-wrapper">
              <iframe
                src="https://voiceoasis.azurewebsites.net/"
                title="Embedded Content"
                className="embedded-frame"
                allow="microphone; camera *"
                allowFullScreen
              />
            </div>
          </div>
        </div>
      </div>
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-text">START Hack 2025</div>
          <div className="footer-logos">
            <img src={startGlobalLogo} alt="START Global Logo" className="start-global-logo" />
            <div className="helbling-logos">
              <img src={helblingLogo} alt="Helbling Logo" className="footer-logo" />
              <svg className="helbling-logo" xmlns="http://www.w3.org/2000/svg" width="159" height="38" viewBox="1 0.6 738 174">
                <path fill="rgb(15 60 147)" fillRule="evenodd" d="M705.7 37.4v19.9c-3-7.5-8.5-13.8-15-17.6a45.3 45.3 0 0 0-53.5 8.5 50.2 50.2 0 0 0-12.6 34.4c-.6 13.9 2.9 28 12 38.4 16.7 19.7 53.1 19.3 65.4-5.1v8.4c0 6.3-1.1 11.5-3.4 15.6-9.2 16.8-40.7 11.3-57.5 9.9v20.8c4 .6 8.5 1.1 13.3 1.4 24.3 1.4 52.8 1.2 65.1-20.6 4-7.3 6.1-16.8 6.1-28.6V55.9H739V37.4h-33.3zm-4.1 49.2a29 29 0 0 1-3.5 14.6 24 24 0 0 1-9.4 9.3 27 27 0 0 1-13 3.2c-5.2 0-9.8-1.2-13.8-3.6-4-2.4-7.1-5.8-9.4-10.1a33.1 33.1 0 0 1-3.4-15.5c0-5.9 1.1-11.1 3.4-15.6s5.4-7.9 9.4-10.3a28.4 28.4 0 0 1 26.4-.8 23.8 23.8 0 0 1 13.2 22.7v6.1zm-97-7.9v55.4h-24.7V76.5c.1-11.5-7.4-20.1-19.1-19.9-12.1-.2-20.5 8.6-20.3 20.7v56.8h-24.8V37.4h19.6v23.9a34 34 0 0 1 13.8-21.9c12.8-8.3 37.7-6.5 46.6 6.1 5.9 7.4 8.9 18.5 8.9 33.2zM488.4 37.4v96.7h-24.8V55.9h-13.5V37.4h38.3zm-24.7-12.5h24.8V.1h-24.8v24.8zM429.9 4.1v130h-24.8V22.6H393V4.1h36.9zm-50.3 60c-4-11.6-12-21-23.1-26-18-8.1-42.1-3.3-51.8 15.1V4.1H280v130h19.6v-20.9c3.1 8.1 8.9 15 16.2 19.1a45.7 45.7 0 0 0 54.5-9.4 53 53 0 0 0 12.5-35.5c.2-8.1-.6-16.4-3.2-23.3zm-25.1 37.4c-11.7 25.4-50.8 16.4-50.5-11.6v-7.1c0-5.8 1.3-10.8 3.8-14.9 5-8.2 13.8-12.8 23.4-12.7 24.2-.2 31.8 27.4 23.3 46.3zM251.9 4.1v130h-24.8V22.6H215V4.1h36.9zm-47.3 78.7c.1-27.3-18.6-49.5-47.5-48.8a46.4 46.4 0 0 0-36.3 15.4 54.8 54.8 0 0 0-12.3 38c0 12.3 4.3 24.9 12.4 34.3 4.1 4.8 9.3 8.6 15.5 11.5 11.9 5.6 31 5.7 43.1.2a40 40 0 0 0 23.6-29h-23c-2.9 8.7-12.1 12.8-21.9 12.6-15.8.3-24.6-10.4-26.2-25.1h72.7v-9.1zm-61.4-24.5c7-5 20.5-5.1 27.4-.3 6 3.9 9.3 11.1 10.3 18.5h-48.6a27 27 0 0 1 10.9-18.2zM88.9 78.9v55.2H64.1V76.5C64.3 65 56.5 56.4 45 56.6c-12.3-.1-20.5 9.3-20.3 21.6v55.9H0V4.1h24.8v49.8C30.1 40.6 40.6 34 55.3 34.4c25.4.3 33.4 19.4 33.6 44.5z" clipRule="evenodd" />
              </svg>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
