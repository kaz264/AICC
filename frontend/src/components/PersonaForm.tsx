import { useState } from 'react';
import { Persona, PersonaCreate, createPersona, updatePersona } from '../api/client';

interface Props {
  persona: Persona | null;
  onSaved: () => void;
}

const defaultValues: PersonaCreate = {
  name: '',
  system_prompt: '',
  greeting_message: '안녕하세요, 무엇을 도와드릴까요?',
  language: 'ko-KR',
  stt_provider: 'google',
  tts_provider: 'google',
  tts_voice_id: 'ko-KR-Neural2-C',
  llm_model: 'claude-sonnet-4-20250514',
  vad_sensitivity: 0.5,
  interrupt_threshold_ms: 800,
  filler_enabled: true,
  knowledge_base_id: null,
};

export default function PersonaForm({ persona, onSaved }: Props) {
  const [form, setForm] = useState<PersonaCreate>(
    persona
      ? {
          name: persona.name,
          system_prompt: persona.system_prompt,
          greeting_message: persona.greeting_message,
          language: persona.language,
          stt_provider: persona.stt_provider,
          tts_provider: persona.tts_provider,
          tts_voice_id: persona.tts_voice_id,
          llm_model: persona.llm_model,
          vad_sensitivity: persona.vad_sensitivity,
          interrupt_threshold_ms: persona.interrupt_threshold_ms,
          filler_enabled: persona.filler_enabled,
          knowledge_base_id: persona.knowledge_base_id,
        }
      : defaultValues
  );
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      if (persona) {
        await updatePersona(persona.id, form);
      } else {
        await createPersona(form);
      }
      onSaved();
    } catch (err) {
      console.error('저장 실패:', err);
    } finally {
      setSaving(false);
    }
  };

  const set = (field: keyof PersonaCreate, value: unknown) =>
    setForm((prev) => ({ ...prev, [field]: value }));

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow p-6 space-y-4">
      <h3 className="text-lg font-semibold">
        {persona ? '페르소나 수정' : '새 페르소나 생성'}
      </h3>

      {/* 이름 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">이름</label>
        <input
          type="text"
          value={form.name}
          onChange={(e) => set('name', e.target.value)}
          placeholder="예: 보험 상담사"
          required
          className="w-full border border-gray-300 rounded-lg px-3 py-2"
        />
      </div>

      {/* 시스템 프롬프트 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">시스템 프롬프트</label>
        <textarea
          value={form.system_prompt}
          onChange={(e) => set('system_prompt', e.target.value)}
          rows={6}
          required
          className="w-full border border-gray-300 rounded-lg px-3 py-2 font-mono text-sm"
        />
      </div>

      {/* 인사말 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">인사말</label>
        <input
          type="text"
          value={form.greeting_message}
          onChange={(e) => set('greeting_message', e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-3 py-2"
        />
      </div>

      {/* TTS 설정 */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">TTS 제공자</label>
          <select
            value={form.tts_provider}
            onChange={(e) => set('tts_provider', e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2"
          >
            <option value="google">Google Cloud TTS</option>
            <option value="elevenlabs">ElevenLabs</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">음성 ID</label>
          <input
            type="text"
            value={form.tts_voice_id}
            onChange={(e) => set('tts_voice_id', e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2"
          />
        </div>
      </div>

      {/* VAD 설정 */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            VAD 민감도: {form.vad_sensitivity}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={form.vad_sensitivity}
            onChange={(e) => set('vad_sensitivity', parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            침묵 감지 (ms)
          </label>
          <input
            type="number"
            value={form.interrupt_threshold_ms}
            onChange={(e) => set('interrupt_threshold_ms', parseInt(e.target.value))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2"
          />
        </div>
      </div>

      {/* 추임새 */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="filler"
          checked={form.filler_enabled}
          onChange={(e) => set('filler_enabled', e.target.checked)}
        />
        <label htmlFor="filler" className="text-sm text-gray-700">
          자연스러운 추임새 사용 (네~, 음...)
        </label>
      </div>

      {/* 지식베이스 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          지식베이스 ID (선택)
        </label>
        <input
          type="text"
          value={form.knowledge_base_id || ''}
          onChange={(e) => set('knowledge_base_id', e.target.value || null)}
          placeholder="예: insurance_kb"
          className="w-full border border-gray-300 rounded-lg px-3 py-2"
        />
      </div>

      <button
        type="submit"
        disabled={saving}
        className="w-full bg-blue-600 text-white py-2.5 rounded-lg font-medium hover:bg-blue-700 transition disabled:opacity-50"
      >
        {saving ? '저장 중...' : persona ? '수정' : '생성'}
      </button>
    </form>
  );
}
