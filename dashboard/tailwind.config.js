/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        nexus: {
          bg: "color-mix(in srgb, var(--bg-primary) calc(<alpha-value> * 100%), transparent)",
          surface: "color-mix(in srgb, var(--bg-surface) calc(<alpha-value> * 100%), transparent)",
          "surface-hover": "color-mix(in srgb, var(--bg-surface-hover) calc(<alpha-value> * 100%), transparent)",
          border: "color-mix(in srgb, var(--border) calc(<alpha-value> * 100%), transparent)",
          "text-primary": "color-mix(in srgb, var(--text-primary) calc(<alpha-value> * 100%), transparent)",
          "text-secondary": "color-mix(in srgb, var(--text-secondary) calc(<alpha-value> * 100%), transparent)",
          "text-muted": "color-mix(in srgb, var(--text-muted) calc(<alpha-value> * 100%), transparent)",
          "accent-blue": "color-mix(in srgb, var(--accent-blue) calc(<alpha-value> * 100%), transparent)",
          "accent-purple": "color-mix(in srgb, var(--accent-purple) calc(<alpha-value> * 100%), transparent)",
          "stress-high": "color-mix(in srgb, var(--stress-high) calc(<alpha-value> * 100%), transparent)",
          "stress-med": "color-mix(in srgb, var(--stress-med) calc(<alpha-value> * 100%), transparent)",
          "stress-low": "color-mix(in srgb, var(--stress-low) calc(<alpha-value> * 100%), transparent)",
          neutral: "color-mix(in srgb, var(--neutral) calc(<alpha-value> * 100%), transparent)",
          confidence: "color-mix(in srgb, var(--confidence) calc(<alpha-value> * 100%), transparent)",
          engagement: "color-mix(in srgb, var(--engagement) calc(<alpha-value> * 100%), transparent)",
          alert: "color-mix(in srgb, var(--alert) calc(<alpha-value> * 100%), transparent)",
          "agent-voice": "color-mix(in srgb, var(--agent-voice) calc(<alpha-value> * 100%), transparent)",
          "agent-language": "color-mix(in srgb, var(--agent-language) calc(<alpha-value> * 100%), transparent)",
          "agent-facial": "color-mix(in srgb, var(--agent-facial) calc(<alpha-value> * 100%), transparent)",
          "agent-body": "color-mix(in srgb, var(--agent-body) calc(<alpha-value> * 100%), transparent)",
          "agent-gaze": "color-mix(in srgb, var(--agent-gaze) calc(<alpha-value> * 100%), transparent)",
          "agent-conversation": "color-mix(in srgb, var(--agent-conversation) calc(<alpha-value> * 100%), transparent)",
          "agent-fusion": "color-mix(in srgb, var(--agent-fusion) calc(<alpha-value> * 100%), transparent)",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        'nexus': '0 1px 3px var(--card-shadow)',
        'nexus-lg': '0 4px 12px var(--card-shadow)',
      },
    },
  },
  plugins: [],
};
