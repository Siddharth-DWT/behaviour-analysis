import { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "./contexts/AuthContext";
import { setAccessToken } from "./api/client";
import Layout from "./components/Layout";
import SessionList from "./pages/SessionList";
import SessionDetail from "./pages/SessionDetail";
import ReportView from "./pages/ReportView";
import Login from "./pages/Login";
import Signup from "./pages/Signup";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-nexus-bg">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-nexus-accent-blue" />
          <span className="text-xs text-nexus-text-muted">Loading...</span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-nexus-bg">
        <Loader2 className="h-6 w-6 animate-spin text-nexus-accent-blue" />
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/sessions" replace />;
  }

  return <>{children}</>;
}

export default function App() {
  const { accessToken } = useAuth();

  // Sync access token to the API client module
  useEffect(() => {
    setAccessToken(accessToken);
  }, [accessToken]);

  return (
    <Routes>
      {/* Public routes */}
      <Route
        path="/login"
        element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <PublicRoute>
            <Signup />
          </PublicRoute>
        }
      />

      {/* Protected routes */}
      <Route
        path="/sessions"
        element={
          <ProtectedRoute>
            <Layout>
              <SessionList />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/sessions/:id"
        element={
          <ProtectedRoute>
            <Layout>
              <SessionDetail />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/sessions/:id/report"
        element={
          <ProtectedRoute>
            <Layout>
              <ReportView />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* Default redirect */}
      <Route path="*" element={<Navigate to="/sessions" replace />} />
    </Routes>
  );
}
