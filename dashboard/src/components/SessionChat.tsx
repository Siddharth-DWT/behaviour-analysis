import { useState, useRef, useEffect, useCallback } from "react";
import { getAccessToken } from "../api/client";

// ═══════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: { type: string; text: string; similarity: number }[];
}

interface SessionChatProps {
  sessionId: string;
  meetingType?: string;
}

// ═══════════════════════════════════════════
// SUGGESTIONS BY MEETING TYPE
// ═══════════════════════════════════════════

const SUGGESTIONS: Record<string, string[]> = {
  sales_call: [
    "What were the key moments in this call?",
    "Were there any objections and how were they handled?",
    "Was the prospect genuinely interested?",
    "What commitments were made?",
    "How stressed was each speaker?",
  ],
  internal: [
    "What were the main topics discussed?",
    "Who dominated the conversation?",
    "Were there any tension points?",
    "What decisions or commitments were made?",
    "What was the overall sentiment trajectory?",
  ],
  interview: [
    "How did the candidate perform?",
    "What were the key questions asked?",
    "Were there signs of stress or nervousness?",
    "What commitments or next steps were discussed?",
  ],
  default: [
    "What were the key moments in this conversation?",
    "How stressed was each speaker?",
    "What was the overall sentiment trajectory?",
    "Who dominated the conversation?",
    "Were there signs of incongruence between voice and language?",
    "What commitments were made?",
  ],
};

// ═══════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════

export default function SessionChat({ sessionId, meetingType }: SessionChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const suggestions = SUGGESTIONS[meetingType || ""] || SUGGESTIONS.default;

  // Load persisted chat history on mount
  useEffect(() => {
    const token = getAccessToken();
    if (!token) return;
    fetch(`/api/sessions/${sessionId}/chat`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.messages?.length) {
          setMessages(data.messages);
        }
      })
      .catch(() => {});
  }, [sessionId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || loading) return;
      const userMsg: ChatMessage = { role: "user", content: text.trim() };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);

      try {
        const token = getAccessToken();
        const res = await fetch(`/api/sessions/${sessionId}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            question: text.trim(),
            history: messages.slice(-4).map((m) => ({ role: m.role, content: m.content })),
          }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || `Error ${res.status}`);
        }

        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.answer, sources: data.sources },
        ]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unexpected error";
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${msg}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, messages, loading]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-sm text-nexus-text-secondary">Try asking:</p>
            {suggestions.map((q, i) => (
              <button
                key={i}
                onClick={() => sendMessage(q)}
                className="block w-full text-left text-sm rounded-lg px-3 py-2
                           bg-nexus-surface border border-nexus-border
                           text-nexus-text-secondary hover:bg-nexus-surface-hover
                           hover:text-nexus-text-primary transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                msg.role === "user"
                  ? "bg-nexus-accent-blue text-white"
                  : "bg-nexus-surface border border-nexus-border text-nexus-text-primary"
              }`}
            >
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 pt-2 border-t border-nexus-border">
                  <p className="text-[10px] text-nexus-text-muted mb-1">Sources:</p>
                  {msg.sources.map((s, j) => (
                    <p key={j} className="text-[10px] text-nexus-text-muted truncate">
                      <span className="inline-block w-16 text-nexus-text-secondary">[{s.type}]</span>{" "}
                      {s.text}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-nexus-surface border border-nexus-border rounded-lg px-4 py-3">
              <div className="flex space-x-1.5">
                <span className="w-2 h-2 rounded-full bg-nexus-accent-blue animate-bounce" />
                <span className="w-2 h-2 rounded-full bg-nexus-accent-blue animate-bounce [animation-delay:0.15s]" />
                <span className="w-2 h-2 rounded-full bg-nexus-accent-blue animate-bounce [animation-delay:0.3s]" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-nexus-border p-3">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this session..."
            disabled={loading}
            className="flex-1 rounded-lg border border-nexus-border bg-nexus-surface px-3 py-2
                       text-sm text-nexus-text-primary placeholder:text-nexus-text-muted
                       focus:outline-none focus:border-nexus-accent-blue disabled:opacity-50"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || loading}
            className="rounded-lg bg-nexus-accent-blue px-4 py-2 text-sm font-medium text-white
                       hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
