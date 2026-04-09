import { Routes, Route, Link, useLocation } from 'react-router-dom';
import AdminDashboard from './pages/AdminDashboard';
import TestCall from './pages/TestCall';

function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">AICC PoC</h1>
          <nav className="flex gap-4">
            <Link
              to="/"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              페르소나 관리
            </Link>
            <Link
              to="/call"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/call'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              테스트 통화
            </Link>
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<AdminDashboard />} />
          <Route path="/call" element={<TestCall />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
