import { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "./contexts/AuthContext";
import { setAccessToken } from "./api/client";
import Layout from "./components/Layout";
import UploadPage from "./pages/UploadPage";
import SessionList from "./pages/SessionList";
import SessionDetail from "./pages/SessionDetail";
import ReportView from "./pages/ReportView";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import VerifyEmail from "./pages/VerifyEmail";
import ProfilePage from "./pages/ProfilePage";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";

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
    return <Navigate to="/upload" replace />;
  }

  return <>{children}</>;
}

export default function App() {
  const { accessToken } = useAuth();

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
      <Route path="/verify-email" element={<VerifyEmail />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />

      {/* ── Primary: Upload ── */}
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
            <Layout>
              <UploadPage />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* Legacy redirect — /sessions/new → /upload */}
      <Route path="/sessions/new" element={<Navigate to="/upload" replace />} />

      {/* ── Sessions ── */}
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

      {/* ── Profile ── */}
      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <Layout>
              <ProfilePage />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* Default → Upload */}
      <Route path="*" element={<Navigate to="/upload" replace />} />
    </Routes>
  );
}
