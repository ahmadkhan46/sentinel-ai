"use client";

import { useEffect, useRef, useState } from "react";

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export interface InferenceEvent {
  asset_id: string;
  rul: number | null;
  is_anomaly: boolean | null;
}

export function useOrgWebSocket(orgId: string | undefined) {
  const [lastEvent, setLastEvent] = useState<InferenceEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!orgId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/${orgId}`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        const data: InferenceEvent = JSON.parse(e.data);
        setLastEvent(data);
      } catch {}
    };

    ws.onerror = () => {};

    return () => {
      ws.close();
    };
  }, [orgId]);

  return lastEvent;
}
