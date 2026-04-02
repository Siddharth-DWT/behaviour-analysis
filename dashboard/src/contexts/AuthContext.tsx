import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  ReactNode,
} from "react";

// ── Types ──

export interface User {
  id: string;
  email: string;
  full_name: string;
  company?: string;
  role: string;
  avatar_url?: string;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (
    email: string,
    password: string,
    fullName: string,
    company?: string
  ) => Promise<void>;
  logout: () => void;
  updateUser: (user: User) => void;
}

const AuthContext = createContext<AuthState>({
  user: null,
  accessToken: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => {},
  signup: async () => {},
  logout: () => {},
  updateUser: () => {},
});

// ── API helpers ──

const API_BASE = "/api";

async function authRequest<T>(
  path: string,
  body: Record<string, unknown>,
  accessToken?: string | null
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(data.detail || `Error ${res.status}`);
  }
  return res.json();
}

// ── Token helpers ──

function getStoredRefreshToken(): string | null {
  return localStorage.getItem("nexus_refresh_token");
}

function setStoredRefreshToken(token: string | null) {
  if (token) {
    localStorage.setItem("nexus_refresh_token", token);
  } else {
    localStorage.removeItem("nexus_refresh_token");
  }
}

function parseJwtExp(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.exp ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}

// ── Provider ──

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearAuth = useCallback(() => {
    setUser(null);
    setAccessToken(null);
    setStoredRefreshToken(null);
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }
  }, []);

  const scheduleRefresh = useCallback(
    (token: string, refreshToken: string) => {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
      const exp = parseJwtExp(token);
      if (!exp) return;

      // Refresh 2 minutes before expiry
      const delay = Math.max(exp - Date.now() - 2 * 60 * 1000, 10_000);
      refreshTimerRef.current = setTimeout(async () => {
        try {
          const data = await authRequest<{
            user: User;
            access_token: string;
            refresh_token: string;
          }>("/auth/refresh", { refresh_token: refreshToken });

          setUser(data.user);
          setAccessToken(data.access_token);
          setStoredRefreshToken(data.refresh_token);
          scheduleRefresh(data.access_token, data.refresh_token);
        } catch {
          clearAuth();
        }
      }, delay);
    },
    [clearAuth]
  );

  const handleAuthResponse = useCallback(
    (data: { user: User; access_token: string; refresh_token: string }) => {
      setUser(data.user);
      setAccessToken(data.access_token);
      setStoredRefreshToken(data.refresh_token);
      scheduleRefresh(data.access_token, data.refresh_token);
    },
    [scheduleRefresh]
  );

  // Restore session on mount (guarded against React StrictMode double-fire)
  const restoreAttempted = useRef(false);
  useEffect(() => {
    if (restoreAttempted.current) return;
    restoreAttempted.current = true;

    const storedRefresh = getStoredRefreshToken();
    if (!storedRefresh) {
      setIsLoading(false);
      return;
    }

    authRequest<{
      user: User;
      access_token: string;
      refresh_token: string;
    }>("/auth/refresh", { refresh_token: storedRefresh })
      .then((data) => {
        handleAuthResponse(data);
      })
      .catch(() => {
        clearAuth();
      })
      .finally(() => {
        setIsLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const data = await authRequest<{
        user: User;
        access_token: string;
        refresh_token: string;
      }>("/auth/login", { email, password });
      handleAuthResponse(data);
    },
    [handleAuthResponse]
  );

  const signup = useCallback(
    async (
      email: string,
      password: string,
      fullName: string,
      company?: string
    ) => {
      const data = await authRequest<{
        user: User;
        access_token?: string;
        refresh_token?: string;
        requires_verification?: boolean;
        message?: string;
      }>("/auth/signup", {
        email,
        password,
        full_name: fullName,
        company: company || undefined,
      });

      if (data.requires_verification) {
        // Don't auto-login — throw a special error the Signup page will catch
        throw Object.assign(new Error(data.message || "Please verify your email"), {
          requiresVerification: true,
          email,
        });
      }

      // Normal flow (email not configured — auto-verified)
      handleAuthResponse(data as { user: User; access_token: string; refresh_token: string });
    },
    [handleAuthResponse]
  );

  const logout = useCallback(async () => {
    const refreshToken = getStoredRefreshToken();
    if (refreshToken && accessToken) {
      try {
        await authRequest(
          "/auth/logout",
          { refresh_token: refreshToken },
          accessToken
        );
      } catch {
        // Ignore logout errors
      }
    }
    clearAuth();
  }, [accessToken, clearAuth]);

  const updateUser = useCallback((updatedUser: User) => {
    setUser(updatedUser);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        accessToken,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        logout,
        updateUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
