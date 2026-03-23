export interface ContentTypeConfig {
  label: string;
  roles: string[];
  entityFields: string[];
  speakerStats: Record<string, string[]>;
}

export const CONTENT_TYPES: Record<string, ContentTypeConfig> = {
  sales_call: {
    label: "Sales Call",
    roles: ["Seller", "Prospect"],
    entityFields: ["objections", "buying_signals", "commitments", "sales_stages"],
    speakerStats: { Seller: ["persuasion_count"], Prospect: ["buying_signal_count"] },
  },
  client_meeting: {
    label: "Client Meeting",
    roles: ["Account Manager", "Client"],
    entityFields: ["action_items", "decisions", "satisfaction_indicators", "risk_flags"],
    speakerStats: { Client: ["satisfaction"], "Account Manager": ["action_items_owned"] },
  },
  internal: {
    label: "Internal Meeting",
    roles: ["Facilitator", "Participant"],
    entityFields: ["action_items", "decisions"],
    speakerStats: { all: ["talk_time_pct"] },
  },
  interview: {
    label: "Interview",
    roles: ["Interviewer", "Candidate"],
    entityFields: ["questions_asked", "candidate_strengths", "candidate_concerns"],
    speakerStats: { Candidate: ["confidence"], Interviewer: ["questions_asked_count"] },
  },
  podcast: {
    label: "Podcast",
    roles: ["Host", "Guest"],
    entityFields: ["topics", "key_terms"],
    speakerStats: { all: ["talk_time_pct"] },
  },
  debate: {
    label: "Debate",
    roles: ["Speaker A", "Speaker B"],
    entityFields: ["topics", "key_terms"],
    speakerStats: { all: ["talk_time_pct", "stress_avg"] },
  },
};

export function getConfig(contentType: string): ContentTypeConfig {
  return CONTENT_TYPES[contentType] || CONTENT_TYPES.sales_call;
}
