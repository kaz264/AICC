import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

// ── Persona API ──

export interface Persona {
  id: string;
  name: string;
  system_prompt: string;
  greeting_message: string;
  language: string;
  stt_provider: string;
  tts_provider: string;
  tts_voice_id: string;
  llm_model: string;
  vad_sensitivity: number;
  interrupt_threshold_ms: number;
  filler_enabled: boolean;
  knowledge_base_id: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export type PersonaCreate = Omit<Persona, 'id' | 'created_at' | 'updated_at'>;
export type PersonaUpdate = Partial<PersonaCreate>;

export const getPersonas = () => api.get<Persona[]>('/personas/');
export const getPersona = (id: string) => api.get<Persona>(`/personas/${id}`);
export const createPersona = (data: PersonaCreate) => api.post<Persona>('/personas/', data);
export const updatePersona = (id: string, data: PersonaUpdate) => api.put<Persona>(`/personas/${id}`, data);
export const deletePersona = (id: string) => api.delete(`/personas/${id}`);

// ── Call API ──

export interface StartCallResponse {
  call_id: string;
  room_url: string;
  token: string;
}

export const startCall = (persona_id: string) =>
  api.post<StartCallResponse>('/calls/start', { persona_id });

export const getCallLogs = () => api.get('/calls/logs');
