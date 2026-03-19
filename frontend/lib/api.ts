const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

// ── Types ────────────────────────────────────────────────────────────────────

export interface TokenOut {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface UserOut {
  id: string;
  org_id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  last_login: string | null;
  created_at: string;
}

export interface AssetOut {
  id: string;
  org_id: string;
  name: string;
  asset_type: string;
  serial_number: string | null;
  location: string | null;
  description: string | null;
  status: "operational" | "warning" | "critical" | "offline";
  health_index: number | null;
  last_rul: number | null;
  last_inference_at: string | null;
  model_name: string | null;
  model_version: string | null;
  created_at: string;
  updated_at: string;
}

export interface AlertOut {
  id: string;
  asset_id: string;
  org_id: string;
  severity: "info" | "warning" | "critical";
  alert_type: string;
  title: string;
  message: string;
  rul_at_alert: number | null;
  anomaly_score_at_alert: number | null;
  status: "open" | "acknowledged" | "resolved";
  acknowledged_by: string | null;
  acknowledged_at: string | null;
  resolved_at: string | null;
  created_at: string;
}

export interface InferenceOut {
  id: string;
  asset_id: string;
  model_name: string;
  model_version: string;
  inferred_at: string;
  cycle: number | null;
  anomaly_score: number | null;
  is_anomaly: boolean | null;
  anomaly_threshold: number | null;
  rul_prediction: number | null;
  health_index: number | null;
  shap_values: Record<string, number> | null;
  feature_importance: Record<string, number> | null;
}

export interface FleetSummary {
  total_assets: number;
  operational: number;
  warning: number;
  critical: number;
  offline: number;
  open_alerts: number;
  avg_health_index: number | null;
  avg_rul: number | null;
}

export interface AssetTrend {
  asset_id: string;
  asset_name: string;
  timestamps: string[];
  rul_values: (number | null)[];
  health_index_values: (number | null)[];
  anomaly_scores: (number | null)[];
}

// ── Auth ─────────────────────────────────────────────────────────────────────

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

export function saveTokens(tokens: TokenOut): void {
  localStorage.setItem("access_token", tokens.access_token);
  localStorage.setItem("refresh_token", tokens.refresh_token);
}

export function clearTokens(): void {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── API calls ────────────────────────────────────────────────────────────────

export const api = {
  // Auth
  login: (email: string, password: string) =>
    request<TokenOut>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  me: () => request<UserOut>("/auth/me"),

  // Assets
  listAssets: () => request<AssetOut[]>("/assets"),
  getAsset: (id: string) => request<AssetOut>(`/assets/${id}`),

  // Inference
  triggerInference: (assetId: string) =>
    request<InferenceOut>(`/assets/${assetId}/inference`, { method: "POST", body: "{}" }),

  listInference: (assetId: string) =>
    request<InferenceOut[]>(`/assets/${assetId}/inference`),

  // Alerts
  listAlerts: (params?: { status?: string; severity?: string; asset_id?: string }) => {
    const qs = new URLSearchParams(params as Record<string, string>).toString();
    return request<AlertOut[]>(`/alerts${qs ? "?" + qs : ""}`);
  },
  acknowledgeAlert: (id: string) =>
    request<AlertOut>(`/alerts/${id}/acknowledge`, { method: "POST", body: "{}" }),
  resolveAlert: (id: string) =>
    request<AlertOut>(`/alerts/${id}/resolve`, { method: "POST", body: "{}" }),

  // Analytics
  fleetSummary: () => request<FleetSummary>("/analytics/fleet"),
  assetTrend: (id: string) => request<AssetTrend>(`/analytics/assets/${id}/trend`),
};
