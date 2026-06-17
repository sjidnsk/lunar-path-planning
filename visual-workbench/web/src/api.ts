export async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchRawText(artifactId: string): Promise<string> {
  const response = await fetch(`/api/artifacts/${artifactId}/raw`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.text();
}

export async function fetchRawJson<T>(artifactId: string): Promise<T> {
  const raw = await fetchRawText(artifactId);
  return JSON.parse(raw) as T;
}

export async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const parsed = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(String(parsed.detail ?? response.statusText));
  }
  return parsed as T;
}
