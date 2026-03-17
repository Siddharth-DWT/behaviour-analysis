import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import SessionList from "./pages/SessionList";
import SessionDetail from "./pages/SessionDetail";
import ReportView from "./pages/ReportView";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/sessions" replace />} />
        <Route path="/sessions" element={<SessionList />} />
        <Route path="/sessions/:id" element={<SessionDetail />} />
        <Route path="/sessions/:id/report" element={<ReportView />} />
      </Routes>
    </Layout>
  );
}
