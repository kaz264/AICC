import { useEffect, useState } from 'react';
import type { Persona } from '../api/client';
import { getPersonas, startCall } from '../api/client';
import CallInterface from '../components/CallInterface';

export default function TestCall() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [callInfo, setCallInfo] = useState<{
    roomUrl: string;
    token: string;
    callId: string;
  } | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getPersonas().then((res) => {
      setPersonas(res.data);
      if (res.data.length > 0) setSelectedId(res.data[0].id);
    });
  }, []);

  const handleStartCall = async () => {
    if (!selectedId) return;
    setConnecting(true);
    setError('');
    try {
      const res = await startCall(selectedId);
      setCallInfo({
        roomUrl: res.data.room_url,
        token: res.data.token,
        callId: res.data.call_id,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '통화 시작 실패';
      setError(msg);
    } finally {
      setConnecting(false);
    }
  };

  const handleEndCall = () => {
    setCallInfo(null);
  };

  const selectedPersona = personas.find((p) => p.id === selectedId);

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">테스트 통화</h2>

      {!callInfo ? (
        <div className="bg-white rounded-xl shadow p-6 space-y-6">
          {/* 페르소나 선택 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              페르소나 선택
            </label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
            >
              {personas.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* 선택된 페르소나 미리보기 */}
          {selectedPersona && (
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <p className="text-sm text-gray-600">
                <span className="font-medium">인사말:</span>{' '}
                {selectedPersona.greeting_message}
              </p>
              <p className="text-sm text-gray-600">
                <span className="font-medium">TTS:</span>{' '}
                {selectedPersona.tts_provider} / {selectedPersona.tts_voice_id}
              </p>
              <p className="text-sm text-gray-600">
                <span className="font-medium">LLM:</span>{' '}
                {selectedPersona.llm_model}
              </p>
            </div>
          )}

          {error && (
            <p className="text-red-600 text-sm">{error}</p>
          )}

          {/* 통화 시작 버튼 */}
          <button
            onClick={handleStartCall}
            disabled={connecting || !selectedId}
            className="w-full bg-green-600 text-white py-3 rounded-lg text-lg font-medium hover:bg-green-700 transition disabled:opacity-50"
          >
            {connecting ? '연결 중...' : '통화 시작'}
          </button>
        </div>
      ) : (
        <CallInterface
          roomUrl={callInfo.roomUrl}
          token={callInfo.token}
          personaName={selectedPersona?.name || 'AI'}
          onEnd={handleEndCall}
        />
      )}
    </div>
  );
}
