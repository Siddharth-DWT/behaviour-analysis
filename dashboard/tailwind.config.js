/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        nexus: {
          bg: "#0F1117",
          surface: "#1A1D27",
          "surface-hover": "#242836",
          border: "#2D3348",
          "text-primary": "#E8ECF4",
          "text-secondary": "#8B93A7",
          "text-muted": "#565E73",
          "accent-blue": "#4F8BFF",
          "accent-purple": "#8B5CF6",
          "stress-high": "#EF4444",
          "stress-med": "#F59E0B",
          "stress-low": "#22C55E",
          neutral: "#6B7280",
          confidence: "#3B82F6",
          engagement: "#10B981",
          alert: "#F97316",
          "agent-voice": "#4F8BFF",
          "agent-language": "#8B5CF6",
          "agent-facial": "#F59E0B",
          "agent-body": "#10B981",
          "agent-gaze": "#EC4899",
          "agent-conversation": "#06B6D4",
          "agent-fusion": "#F97316",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
